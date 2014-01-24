#include <pcl/io/pcd_io.h>
#include <pcl/common/time.h>
#include <pcl/console/parse.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/pcl_base.h>

#include <pcl/octree/octree_pointcloud_sequential.h>

#include <boost/regex.hpp>
#include <boost/format.hpp>
#include <set>

//#include <pcl/octree/impl/octree_pointcloud.hpp>
//#include <pcl/octree/impl/octree_base.hpp>
//#include <pcl/octree/impl/octree_iterator.hpp>
//#include <pcl/octree/impl/octree_pointcloud_sequential.hpp>
//#include <pcl/octree/impl/octree_pointcloud_adjacency.hpp>

typedef pcl::PointXYZRGBA PointT;
typedef typename pcl::PointCloud<PointT> PointCloudT;
typedef pcl::octree::SequentialVoxelData< PointT> SeqVoxelDataT;
typedef pcl::octree::OctreePointCloudAdjacencyContainer<PointT, SeqVoxelDataT> AdjSeqContainerT;
typedef typename pcl::octree::OctreePointCloudSequential<PointT, AdjSeqContainerT> OctreeSequentialT;
typedef typename OctreeSequentialT::LeafKeyVectorT LeafKeyVectorT;


//template class pcl::octree::OctreePointCloudAdjacencyContainer<PointT, SeqVoxelDataT >;
//template class pcl::octree::OctreePointCloudSequential<PointT, AdjSeqContainerT>;

using namespace pcl;
using namespace pcl::io;
using namespace pcl::console;

bool pause_playback = true;
bool show_original = true;
bool show_sequential = true;
bool show_new = false;

void 
keyboard_callback (const pcl::visualization::KeyboardEvent& event, void*)
{
  if (event.getKeyCode () == '0')
    pause_playback = !pause_playback;
  else if (event.getKeyCode () == '1')
    show_original = !show_original;
  else if (event.getKeyCode () == '2')
    show_sequential = !show_sequential;
  else if (event.getKeyCode () == '3')
    show_new = !show_new;
}

int
main (int argc, char ** argv)
{
  if (argc < 2)
  {
    pcl::console::print_info ("\n \n Syntax is: %s {-p <pcd-file-name-format> -v <voxel resolution>}\n" 
    "Format should be in standard wildcard format (e.g. \"*.pcd\") MUST HAVE QUOTES \n"
    "Program will iterate through sorted list of all pcd files matching this format"
    , argv[0]);
    return (1);
  }
  
  
  bool pcd_file_specified = pcl::console::find_switch (argc, argv, "-p");
  std::string pcd_path_string1;
  if (!pcd_file_specified)
  {
    cout << "No cloud specified!\n";
    return (1);
  }else
  {
    pcl::console::parse (argc,argv,"-p",pcd_path_string1);
  }
  
  float voxel_resolution = 0.01f;
  bool voxel_res_specified = pcl::console::find_switch (argc, argv, "-v");
  if (voxel_res_specified)
    pcl::console::parse (argc, argv, "-v", voxel_resolution);
  
  namespace fs = boost::filesystem;
  cout<< "Loading pointcloud, using format "<< pcd_path_string1<< std::endl;
  fs::path pcd_path1(pcd_path_string1);
  
  std::set<boost::filesystem::path> input_files_original;
  
  cout << "Directory containing cloud files=" << pcd_path1.parent_path ()<<"\n";
  cout << "Filename match string 1=" << pcd_path1.filename ()<<"\n";
  
  std::string wildcard_string_base = pcd_path1.filename (). string ();
  boost::replace_all(wildcard_string_base, "*", ".*");
  std::string wildcard_string_original = ".*_pc.pcd$";
  
  if (fs::exists (pcd_path1.parent_path ()) && fs::is_directory (pcd_path1.parent_path () ))
  {
    boost::regex pattern_original(wildcard_string_original);
    fs::directory_iterator end_iter;
    for( fs::directory_iterator dir_iter(pcd_path1.parent_path ()) ; dir_iter != end_iter ; ++dir_iter)
    {
      //std::cout << "Attempting to match "<<dir_iter->path ().string ()<<"\n";
      if (fs::is_regular_file(dir_iter->status()) )
      {
        //std::cout << dir_iter->path ().string () << "\n";
        if (boost::regex_match (dir_iter->path ().string (), pattern_original))
        {
          input_files_original.insert( *dir_iter );
          cout << "Adding "<<dir_iter->path ().string ()<<" to original\n";
        }
      }
    }
  }
  else
  {
    cout << "Wildcard string original="<<wildcard_string_original<<endl;
    cout << pcd_path1 << " does not exist\n";
    return (1);
  }
  cout << "Found "<<input_files_original.size() << " files matching "<<wildcard_string_original<<"\n";
  
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->registerKeyboardCallback(keyboard_callback, 0);
  
  
  PointCloudT::Ptr original (new PointCloudT ());
  PointCloudT::Ptr voxel_centroid_cloud (new PointCloudT ());
  PointCloudT::Ptr new_voxels (new PointCloudT ());
  
  std::set<fs::path>::iterator file_itr_original = input_files_original.begin ();
  
  StopWatch timer;
  
  int num_frames = 1;
  
  loadPCDFile (file_itr_original->native (), *original);
  //Fix for old pcd files that were saved incorrectly
  original->sensor_orientation_.x() = 0;
  
  OctreeSequentialT seq_tree (voxel_resolution);
  //seq_tree.setTransformFunction (boost::bind (transformFunction, _1));
  seq_tree.setDifferenceFunction (boost::bind (&OctreeSequentialT::SeqVoxelDataDiff, _1));
  seq_tree.setDifferenceThreshold (0.2);
  seq_tree.setOcclusionTestInterval (0.1f);
  seq_tree.setNumberOfThreads (4);
  
  seq_tree.setInputCloud (original);
  seq_tree.addPointsFromInputCloud();
  voxel_centroid_cloud->resize (seq_tree.size ());
  typename LeafKeyVectorT::iterator leaf_itr = seq_tree.begin ();
  typename PointCloudT::iterator cloud_itr = voxel_centroid_cloud->begin ();
  for ( ; leaf_itr != seq_tree.end (); ++leaf_itr, ++cloud_itr)
  {
    (leaf_itr->first)->getData().getPoint (*cloud_itr);
  }
  double t_seq = 0;
  double t_temp, t_temp2;
  for ( ;file_itr_original != input_files_original.end (); )
  {
    double t_start = timer.getTime ();
    if (!pause_playback)
    {
      // cout << "----------------------Starting new frame ("<<num_frames<<") ------------------\n";
      loadPCDFile (file_itr_original->native (), *original);
      original->sensor_orientation_.x() = 0;
      
      t_temp = timer.getTime();
      seq_tree.setInputCloud (original);
      seq_tree.addPointsFromInputCloud();
      t_temp2 = timer.getTime();
      t_seq += t_temp2 - t_temp;
      if (num_frames %10 ==0)
        std::cout << "Seq time="<<t_seq / num_frames << " ms (avg)\n";
      voxel_centroid_cloud->resize (seq_tree.size ());
      typename LeafKeyVectorT::iterator leaf_itr = seq_tree.begin ();
      typename PointCloudT::iterator cloud_itr = voxel_centroid_cloud->begin ();
      for ( ; leaf_itr != seq_tree.end (); ++leaf_itr, ++cloud_itr)
      {
        (leaf_itr->first)->getData().getPoint (*cloud_itr);
      }
      
      
      ++file_itr_original;    
      num_frames++;
    }
    
    if (show_original)
    {
      pcl::visualization::PointCloudColorHandlerRGBField<PointT> orig(original);
      if (!viewer->updatePointCloud (original, orig, "original"))
        viewer->addPointCloud (original, orig, "original");
      viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,2.0, "original");
    }
    else
    {
      viewer->removePointCloud("original");
    }
    
    if (show_sequential)
    {
      pcl::visualization::PointCloudColorHandlerRGBField<PointT> seq(voxel_centroid_cloud);
      if (!viewer->updatePointCloud (voxel_centroid_cloud, seq, "sequential"))
        viewer->addPointCloud (voxel_centroid_cloud, seq, "sequential");
      viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,4.0, "sequential");
    }
    else
    {
      viewer->removePointCloud ("sequential");
    }
    
    if (show_new)
    {
      new_voxels = seq_tree.getNewVoxelCloud ();
      if (!viewer->updatePointCloud (new_voxels, "new_voxels"))
        viewer->addPointCloud (new_voxels, "new_voxels");
      viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,4.0, "new_voxels");
    }
    else
    {
      viewer->removePointCloud ("new_voxels");
    }
    
    double t_spent = timer.getTime () - t_start; 
    int sleep_ms = ((50 - t_spent) > 1) ? (50 - t_spent) : 1;
    viewer->spinOnce (sleep_ms);
    // std::cout << "Framerate = "<< 1000.0 / (timer.getTime () - t_start) << "fps\n";
  }
  return (0);
}