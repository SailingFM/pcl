 
 /*
  * Software License Agreement (BSD License)
  *
  *  Point Cloud Library (PCL) - www.pointclouds.org
  *
  *  All rights reserved.
  *
  *  Redistribution and use in source and binary forms, with or without
  *  modification, are permitted provided that the following conditions
  *  are met:
  *
  *   * Redistributions of source code must retain the above copyright
  *     notice, this list of conditions and the following disclaimer.
  *   * Redistributions in binary form must reproduce the above
  *     copyright notice, this list of conditions and the following
  *     disclaimer in the documentation and/or other materials provided
  *     with the distribution.
  *   * Neither the name of Willow Garage, Inc. nor the names of its
  *     contributors may be used to endorse or promote products derived
  *     from this software without specific prior written permission.
  *
  *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
  *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
  *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
  *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
  *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
  *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
  *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
  *  POSSIBILITY OF SUCH DAMAGE.
  *
  * Author : jpapon@gmail.com
  * Email  : jpapon@gmail.com
  *
  */
 
 #ifndef PCL_SEGMENTATION_SEQUENTIAL_SUPERVOXEL_CLUSTERING_H_
 #define PCL_SEGMENTATION_SEQUENTIAL_SUPERVOXEL_CLUSTERING_H_
 
#include <pcl/segmentation/supervoxel_clustering.h>

 namespace pcl
 {
   /** \brief Supervoxel container class - stores a cluster extracted using supervoxel clustering 
    */
   class SequentialSV : public Supervoxel
   {
    public:
      typedef pcl::PointXYZRGBNormal CentroidT;
      typedef pcl::PointXYZRGBNormal VoxelT;
      typedef boost::shared_ptr<SequentialSV> Ptr;
      typedef boost::shared_ptr<const SequentialSV> ConstPtr;

      using Supervoxel::centroid_;
      using Supervoxel::label_;
      using Supervoxel::voxels_;

      SequentialSV (uint32_t label = 0) :
        Supervoxel (label)
      {  } 

      //! \brief Maps voxel index to measured weight - used by tracking
      std::map <size_t, float> voxel_weight_map_;

    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
   };

  /** \brief NEW MESSAGE
  *  \author Jeremie Papon (jpapon@gmail.com)
  *  \ingroup segmentation
  */
  template <typename PointT>
  class PCL_EXPORTS SequentialSVClustering : public SupervoxelClustering<PointT>
  {
    public:
      typedef typename SequentialSV::CentroidT CentroidT;
      typedef typename SequentialSV::VoxelT VoxelT;
      typedef typename SupervoxelClustering<PointT>::VoxelData VoxelData;

      typedef pcl::octree::OctreePointCloudAdjacencyContainer<PointT, VoxelData> LeafContainerT;
      typedef std::vector <LeafContainerT*> LeafVectorT;
      typedef std::map<uint32_t,typename Supervoxel::Ptr> SupervoxelMapT;
      typedef std::map<uint32_t,typename SequentialSV::Ptr> SequentialSVMapT;

      typedef typename pcl::PointCloud<PointT> PointCloudT;
      typedef typename pcl::PointCloud<VoxelT> VoxelCloudT;
      typedef typename pcl::octree::OctreePointCloudAdjacency<PointT, LeafContainerT> OctreeAdjacencyT;
      typedef typename pcl::octree::OctreePointCloudSearch <PointT> OctreeSearchT;
      typedef typename pcl::search::KdTree<PointT> KdTreeT;

    protected:
      typedef typename SupervoxelClustering<PointT>::SupervoxelHelper SupervoxelHelper;
      friend class SupervoxelClustering<PointT>::SupervoxelHelper;
      typedef typename SupervoxelClustering<PointT>::SeedNHood SeedNHood;
      
      using PCLBase <PointT>::initCompute;
      using PCLBase <PointT>::deinitCompute;
      using PCLBase <PointT>::input_;
      using SupervoxelClustering<PointT>::use_single_camera_transform_;
      using SupervoxelClustering<PointT>::seed_prune_radius_;
      
      using SupervoxelClustering<PointT>::transformFunction;
      using SupervoxelClustering<PointT>::transformFunctionVoxel;
      using SupervoxelClustering<PointT>::prepareForSegmentation;
      using SupervoxelClustering<PointT>::selectInitialSupervoxelSeeds;
      using SupervoxelClustering<PointT>::createHelpersFromSeedIndices;
      using SupervoxelClustering<PointT>::expandSupervoxels;
      using SupervoxelClustering<PointT>::initializeLabelColors;
      using SupervoxelClustering<PointT>::findNeighborMinCurvature;
    public:
      using SupervoxelClustering<PointT>::setSeedPruneRadius;
      using SupervoxelClustering<PointT>::setVoxelResolution;
      using SupervoxelClustering<PointT>::getVoxelResolution;
      using SupervoxelClustering<PointT>::setSeedResolution;
      using SupervoxelClustering<PointT>::getSeedResolution;
      using SupervoxelClustering<PointT>::setColorImportance;
      using SupervoxelClustering<PointT>::setSpatialImportance;
      using SupervoxelClustering<PointT>::setNormalImportance;
      using SupervoxelClustering<PointT>::setIgnoreInputNormals;
      using SupervoxelClustering<PointT>::extract;
      using SupervoxelClustering<PointT>::setInputCloud;
      using SupervoxelClustering<PointT>::getVoxelCentroidCloud;
      using SupervoxelClustering<PointT>::getLabeledVoxelCloud;
      using SupervoxelClustering<PointT>::getLabeledCloud;
      using SupervoxelClustering<PointT>::getSupervoxelAdjacency;

      typedef boost::adjacency_list<boost::setS, boost::setS, boost::undirectedS, uint32_t, float> VoxelAdjacencyList;
      typedef VoxelAdjacencyList::vertex_descriptor VoxelID;
      typedef VoxelAdjacencyList::edge_descriptor EdgeID;

    public:
      /** \brief Constructor that sets default values for member variables. 
        *  \param[in] voxel_resolution The resolution (in meters) of voxels used
        *  \param[in] seed_resolution The average size (in meters) of resulting supervoxels
        *  \param[in] use_single_camera_transform Set to true if point density in cloud falls off with distance from origin (such as with a cloud coming from one stationary camera), set false if input cloud is from multiple captures from multiple locations.
        */
      SequentialSVClustering (float voxel_resolution, float seed_resolution, bool use_single_camera_transform = true, bool prune_close_seeds=true);
      
      /** \brief This destructor destroys the cloud, normals and search method used for
        * finding neighbors. In other words it frees memory.
        */
      virtual
      ~SequentialSVClustering ();
      
      /** \brief This method launches the segmentation algorithm and returns the supervoxels that were
       * obtained during the segmentation.
       * \param[out] supervoxel_clusters A map of labels to pointers to supervoxel structures
       */
      virtual void
      extract (std::map<uint32_t,typename SequentialSV::Ptr > &supervoxel_clusters);
      
      void
      setMinWeight (float min_weight)
      {
        min_weight_ = min_weight;
      }
      
      void 
      setFullExpandLeaves (bool do_full_expansion)
      {
        do_full_expansion_ = do_full_expansion;
      }
      void
      buildVoxelCloud ();
      
      /** \brief This function builds new supervoxels which are conditioned on the voxel_weight_maps contained in supervoxel_clusters 
        */
      void
      extractNewConditionedSupervoxels (SequentialSVMapT &supervoxel_clusters, bool add_new_seeds);
    protected:
      void
      createHelpersFromWeightMaps (SequentialSVMapT &supervoxel_clusters, std::vector<size_t> &existing_seed_indices);
      
      void
      clearOwnersSetCentroids ();
      
      void
      expandSupervoxelsFast ( int depth );
      
      /** \brief This method appends internal supervoxel helpers to the list based on the provided seed points
       *  \param[in] seed_indices Indices of the leaves to use as seeds
       */
      void
      appendHelpersFromSeedIndices (std::vector<size_t> &seed_indices);
      
      /** \brief Constructs the map of supervoxel clusters from the internal supervoxel helpers */
      void
      makeSupervoxels (std::map<uint32_t,typename SequentialSV::Ptr > &supervoxel_clusters);

      /** \brief This selects new leaves to use as supervoxel seeds
       *  \param[out] seed_indices The selected leaf indices
       */
      void
      selectNewSupervoxelSeeds (std::vector<size_t> &existing_seed_indices, std::vector<size_t> &seed_indices);
      
      /** \brief Stores the resolution used in the octree */
      using SupervoxelClustering<PointT>::resolution_;

      /** \brief Stores the resolution used to seed the superpixels */
      using SupervoxelClustering<PointT>::seed_resolution_;

      /** \brief Contains a KDtree for the voxelized cloud */
      using SupervoxelClustering<PointT>::voxel_kdtree_;

      /** \brief Octree Adjacency structure with leaves at voxel resolution */
      using SupervoxelClustering<PointT>::adjacency_octree_;

      /** \brief Contains the Voxelized centroid Cloud */
      using SupervoxelClustering<PointT>::voxel_centroid_cloud_;

      /** \brief Importance of color in clustering */
      using SupervoxelClustering<PointT>::color_importance_;
      /** \brief Importance of distance from seed center in clustering */
      using SupervoxelClustering<PointT>::spatial_importance_;
      /** \brief Importance of similarity in normals for clustering */
      using SupervoxelClustering<PointT>::normal_importance_;
      /** \brief Option to ignore normals in input Pointcloud. Defaults to false */
      using SupervoxelClustering<PointT>::ignore_input_normals_; 

      using SupervoxelClustering<PointT>::prune_close_seeds_;

      typedef boost::ptr_list<SupervoxelHelper> HelperListT;
      using SupervoxelClustering<PointT>::supervoxel_helpers_;

      using SupervoxelClustering<PointT>::timer_;
      
      float min_weight_;
      bool do_full_expansion_;
    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
  
 }

 #ifdef PCL_NO_PRECOMPILE
 #include <pcl/segmentation/impl/sequential_supervoxel_clustering.hpp>
 #endif

 #endif //PCL_SEGMENTATION_SEQUENTIAL_SUPERVOXEL_CLUSTERING_H_
