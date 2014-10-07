#ifndef TRACKING_IMPL_STRATIFIED_PARTICLE_FILTER_OMP_HPP_
#define TRACKING_IMPL_STRATIFIED_PARTICLE_FILTER_OMP_HPP_

#include <pcl/tracking/stratified_particle_filter_omp.h>

template <typename PointInT, typename StateT> bool
pcl::tracking::StratifiedParticleFilterOMPTracker<PointInT, StateT>::initCompute ()
{
  if (!Tracker<PointInT, StateT>::initCompute ())
  {
    PCL_ERROR ("[pcl::%s::initCompute] Init failed.\n", getClassName ().c_str ());
    return (false);
  }
  
  if (!particles_ || particles_->points.empty ())
    initParticles (true);
  
  return true;
}


template <typename PointInT, typename StateT> void
pcl::tracking::StratifiedParticleFilterOMPTracker<PointInT, StateT>::weight ()
{
  const std::vector<double> zero_mean (StateT::stateDimension (), 0.0);

  // with motion
  int motion_num = static_cast<int> (particles_->points.size ()) * static_cast<int> (motion_ratio_);
  StateT delta_x = motion_ * delta_t_;
  for (int i = 0; i < motion_num; i++)
  {
    // add noise using gaussian
    particles_->points[i].sample (zero_mean, step_noise_covariance_);
    particles_->points[i] = particles_->points[i] + delta_x;
  }
  
  // no motion
  for ( int i = motion_num; i < particle_num_; i++ )
  {
    // add noise using gaussian
    particles_->points[i].sample (zero_mean, step_noise_covariance_);
  }
  //++frame_count;
  //pcl::StopWatch timer;
  //double t_start = timer.getTime ();

  #ifdef _OPENMP
  #pragma omp parallel for num_threads(threads_) schedule(static, 10)
  #endif
  for (int i = 0; i < particle_num_; i++)
  {
    const Eigen::Affine3f trans = toEigenMatrix (particles_->points[i]);
    particles_->points[i].weight = 0;
    coherences_[i]->compute (ref_, trans, particles_->points[i].weight);
  }
  normalizeWeight ();
 
  /*
  T1 += timer.getTime () - t_start;
  if (frame_count % 10 ==0)
    std::cout<< "==Coherences-   model size="<<this->ref_->size ()<<"  avg t="<<T1/frame_count<<"ms ====="<<std::endl;
  t_start = timer.getTime ();
  */
  
  applyWeights ();
  
  /*
  T2 += timer.getTime () - t_start;
  if (frame_count % 10 ==0)
    std::cout<< "==applyWeights  model size="<<this->ref_->size ()<<"  avg t="<<T2/frame_count<<"ms ====="<<std::endl;
  */
}

template <typename PointInT, typename StateT> void
pcl::tracking::StratifiedParticleFilterOMPTracker<PointInT, StateT>::applyWeights ()
{
  coherences_[0]->clearSupervoxelWeights ();
  //For each particle, have it go through the voxels it sampled and add it's weight to them
  for (int i = 0; i < particle_num_; i++)
  {
    coherences_[i]->applyWeightToStrata (particles_->points[i].weight);
  }
  coherences_[0]->normalizeSupervoxelWeights ();
//  coherences_[0]->printVoxelWeights ();

}

template <typename PointInT, typename StateT> void
pcl::tracking::StratifiedParticleFilterOMPTracker<PointInT, StateT>::update ()
{
  StateT orig_representative = representative_state_;
  representative_state_.zero ();
  representative_state_.weight = 0.0;
  
  for ( size_t i = 0; i < particles_->points.size (); i++)
  {
    StateT p = particles_->points[i];
    representative_state_ = representative_state_ + p * p.weight;
  }
  representative_state_.weight = 1.0f / static_cast<float> (particles_->points.size ());
  motion_ = (representative_state_ - orig_representative) * (1.0 / delta_t_);
  
}

template <typename PointInT, typename StateT> void
pcl::tracking::StratifiedParticleFilterOMPTracker<PointInT, StateT>::setInputCloud (const CloudInTConstPtr &cloud, double t_stamp)
{
  if (last_stamp_ == 0)
    last_stamp_ = t_stamp - 0.1;
  delta_t_ = t_stamp - last_stamp_;
  setInputCloud (cloud);
  last_stamp_ = t_stamp;
  //Set the cloud as target for all coherences
  for (typename std::vector<StratCoherencePtr>::iterator coherence_itr = coherences_.begin(); coherence_itr!= coherences_.end (); ++coherence_itr)
  {
    (*coherence_itr)->setTargetCloud (cloud);
  }
  
  
}

template <typename PointInT, typename StateT> void
pcl::tracking::StratifiedParticleFilterOMPTracker<PointInT, StateT>::resample ()
{
  PointCloudStatePtr S (new PointCloudState);
  S->points.reserve (particle_num_);

  // initializing for sampling without replacement
  std::vector<int> a (particles_->points.size ());
  std::vector<double> q (particles_->points.size ());
  this->genAliasTable (a, q, particles_);
  
  const std::vector<double> zero_mean (StateT::stateDimension (), 0.0);

  PointCloudStatePtr origparticles (new PointCloudState);
  //copyPointCloud (*particles_,*origparticles);
  particles_.swap (origparticles);
  particles_->reserve (origparticles->size ());
  // with motion
  int motion_num = static_cast<int> (origparticles->size () * motion_ratio_);
  StateT delta_x = motion_ * delta_t_;
  for ( int i = 0; i < motion_num; ++i)
  {
    int target_particle_index = sampleWithReplacement (a, q);
    particles_->push_back (origparticles->points[target_particle_index] + delta_x);
    // add noise using gaussian
    particles_->back ().sample (zero_mean, step_noise_covariance_);
    particles_->back ().weight = 1.0/origparticles->size ();
  }
  for ( int i = motion_num; i < origparticles->size (); ++i)
  {
    int target_particle_index = sampleWithReplacement (a, q);
    particles_->push_back(origparticles->points[target_particle_index]);
    // add noise using gaussian
    particles_->back ().sample (zero_mean, step_noise_covariance_);
    particles_->back ().weight = 1.0/origparticles->size ();
  }
}

template <typename PointInT, typename StateT> void
pcl::tracking::StratifiedParticleFilterOMPTracker<PointInT, StateT>::computeTracking ()
{
  for (int i = 0; i < iteration_num_; i++)
  {
    if ( getResult ().weight < resample_likelihood_thr_ )
    {
      resample ();
    }
  
    weight (); 
  
    //TODO
    //Now we have normalized particle weights, we can send scores down to voxels
    applyWeights ();
    
    update ();
    //std::cout <<"Done with update\n";
  }
}

template <typename PointInT, typename StateT> void
pcl::tracking::StratifiedParticleFilterOMPTracker<PointInT, StateT>::updateStrata (std::map<uint32_t,typename SequentialSV::Ptr> &supervoxel_clusters)
{
  for (typename std::vector<StratCoherencePtr>::iterator coherence_itr = coherences_.begin(); coherence_itr!= coherences_.end (); ++coherence_itr)
  {
    (*coherence_itr)->updateStrata (supervoxel_clusters);
  }
  
  std::map<uint32_t,typename SequentialSV::Ptr>::iterator sv_itr;
  std::vector<uint32_t> labels;
  coherences_[0]->getSVLabels (labels);
  typename PointCloud <PointInT>::Ptr new_ref (new PointCloud <PointInT> ());
  //Update model by creating new one.
  for (std::vector<uint32_t>::iterator label_itr = labels.begin () ; label_itr != labels.end (); ++label_itr)
  {
    sv_itr = supervoxel_clusters.find (*label_itr);
    PointCloud<PointInT> temp_rgb;
    pcl::copyPointCloud (*(sv_itr->second)->voxels_, temp_rgb);
    *new_ref += temp_rgb;
  }
  StateT result = this->getResult ();
  Eigen::Affine3f transformation = this->toEigenMatrix (result);
  typename PointCloud <PointInT>::Ptr transed_ref (new CloudInT);
  pcl::transformPointCloud (*new_ref, *transed_ref, transformation.inverse());
  this->setReferenceCloud (new_ref);
}
#define PCL_INSTANTIATE_StratifiedParticleFilterOMPTracker(T,ST) template class PCL_EXPORTS pcl::tracking::StratifiedParticleFilterOMPTracker<T,ST>;

#endif //TRACKING_IMPL_STRATIFIED_PARTICLE_FILTER_OMP_HPP_
