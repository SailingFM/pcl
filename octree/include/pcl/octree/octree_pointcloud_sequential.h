/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2012, Jeremie Papon
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
 *  Author : jpapon@gmail.com
 *  Email  : jpapon@gmail.com
 */

#ifndef PCL_OCTREE_POINTCLOUD_SEQUENTIAL_H_
#define PCL_OCTREE_POINTCLOUD_SEQUENTIAL_H_

#include <pcl/octree/octree_pointcloud_adjacency.h>

namespace pcl
{ 
  namespace octree
  {
    
    template <typename PointT>
    class SequentialVoxelData
    {
      public:
        SequentialVoxelData ():
        xyz_ (0.0f, 0.0f, 0.0f),
        xyz_old_ (0.0f, 0.0f, 0.0f),
        rgb_ (0.0f, 0.0f, 0.0f),
        rgb_old_ (0.0f, 0.0f, 0.0f),
        new_leaf_ (true),
        has_changed_ (false),
        idx_ (0)
        {}
        
        /** \brief Gets the data of in the form of a point
        *  \param[out] point_arg Will contain the point value of the voxeldata
        */  
        void
        getPoint (PointT &point_arg) const;
        
        /** \brief Gets the data of in the form of a point
        *  \return Returns the point value of the voxeldata
        */  
        PointT
        getPoint () const {PointT temp; this->getPoint (temp); return temp; }
        
        bool 
        isNew () const { return new_leaf_; }
        
        void
        setNew (bool new_arg) { new_leaf_ = new_arg; }
        
        bool 
        isChanged () const { return has_changed_; }
        
        void 
        setChanged (bool new_val) { has_changed_ = new_val; }
        
        void
        prepareForNewFrame (const int &points_last_frame)
        {
          new_leaf_ = false;
          has_changed_ = false;
          xyz_old_ = xyz_;
          rgb_old_ = rgb_;
          num_points_ = 0;
        }
        
        void
        revertToLastPoint ()
        {
          xyz_ = xyz_old_;
          rgb_ = rgb_old_;
        }
        
        Eigen::Vector3f xyz_,xyz_old_;
        Eigen::Vector3f rgb_,rgb_old_;
        float distance_;
        int idx_;
        bool has_changed_, new_leaf_;
        int num_points_;
        
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /** \brief @b Octree 
     * 
     */
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template< typename PointT, 
    typename LeafContainerT = OctreePointCloudAdjacencyContainer <PointT, SequentialVoxelData<PointT> >,    
    typename BranchContainerT = OctreeContainerEmpty >
    class OctreePointCloudSequential : public OctreePointCloud< PointT, LeafContainerT, BranchContainerT>
    {
      public:   
        friend class SequentialVoxelData<PointT>;
        typedef SequentialVoxelData<PointT> SeqVoxelDataT;
        typedef OctreePointCloudSequential<PointT, LeafContainerT, BranchContainerT > OctreeSequentialT;
        typedef boost::shared_ptr<OctreeSequentialT> Ptr;
        typedef boost::shared_ptr<const OctreeSequentialT> ConstPtr;
        
        typedef OctreeBase<LeafContainerT, BranchContainerT> OctreeBaseT;
        typedef OctreePointCloud<PointT, LeafContainerT, BranchContainerT,OctreeBaseT > OctreePointCloudT;
        typedef OctreePointCloudAdjacency<PointT, LeafContainerT, BranchContainerT > OctreeAdjacencyT;
        
        typedef typename OctreePointCloudT::LeafNode LeafNode;
        typedef typename OctreePointCloudT::BranchNode BranchNode;
        
        typedef pcl::PointCloud<PointT> PointCloudT;
        
        // iterators are friends
        friend class OctreeIteratorBase<OctreeSequentialT> ;
        friend class OctreeDepthFirstIterator<OctreeSequentialT> ;
        friend class OctreeBreadthFirstIterator<OctreeSequentialT> ;
        friend class OctreeLeafNodeIterator<OctreeSequentialT> ;
        
        // Octree default iterators
        typedef OctreeDepthFirstIterator<OctreeAdjacencyT> Iterator;
        typedef const OctreeDepthFirstIterator<OctreeAdjacencyT> ConstIterator;
        
        Iterator depth_begin (unsigned int max_depth_arg = 0) { return Iterator (this, max_depth_arg); }
        const Iterator depth_end () { return Iterator (); }
        
        // Octree leaf node iterators
        typedef OctreeLeafNodeIterator<OctreeAdjacencyT> LeafNodeIterator;
        typedef const OctreeLeafNodeIterator<OctreeAdjacencyT> ConstLeafNodeIterator;
        
        LeafNodeIterator leaf_begin (unsigned int max_depth_arg = 0) { return LeafNodeIterator (this, max_depth_arg); }
        const LeafNodeIterator leaf_end () { return LeafNodeIterator (); }
        
        typedef typename std::pair<LeafContainerT*, boost::shared_ptr<OctreeKey> > LeafKeyPairT;
        typedef typename std::vector<LeafKeyPairT> LeafKeyVectorT;
        
        // Fast leaf iterators that don't require traversing tree
        typedef typename LeafKeyVectorT::iterator iterator;
        typedef typename LeafKeyVectorT::const_iterator const_iterator;
        
        inline iterator begin () { return (leaf_key_vec_.begin ()); }
        inline iterator end ()   { return (leaf_key_vec_.end ()); }
        
        //! Numbers of leaves
        inline size_t size () const { return leaf_key_vec_.size (); }
        
        /** \brief Constructor.
          *  
          * \param[in] resolution_arg  octree resolution at lowest octree level (voxel size) */
        OctreePointCloudSequential (const double resolution_arg);
        
        
        /** \brief Empty class destructor. */
        virtual ~OctreePointCloudSequential ()
        {
        }
        
        /** \brief Adds points from cloud to the octree  
          * 
          * \note This overrides the addPointsFromInputCloud from the OctreePointCloud class */
        void 
        addPointsFromInputCloud ();
        
        /** \brief Set the difference function which is used to evaluate if a leaf has changed
          * 
          *  \note Generally, this means checking if the color has changed (and possibly the neighbors)
          *  \note There is a default implementation provided for OctreePointCloudAdjacencyContainer<PointT,SequentialVoxelData<PointT>>, but it must be set manually:
          *  \note seq_octree->setDifferenceFunction (boost::bind (&OctreePointCloudSequential::SeqVoxelDataDiff, ptrToSeqOctree, _1));
          *  \param[in] diff_func A boost:function pointer to the difference function to be used. Should return a normalized (0-1) difference value for the given leaf */
        void 
        setDifferenceFunction (boost::function<float (const LeafContainerT* leaf)> diff_func)
        {
          diff_func_ = diff_func;
        }
        
        /** \brief Set the difference threshold for voxel change
          * 
          * \param[in] threshold_arg Sets the threshold value for determining whether a voxel has changed */
        void 
        setDifferenceThreshold (const float threshold_arg) 
        {
          difference_threshold_=threshold_arg;
        }
        
        /** \brief Returns the difference threshold for voxel change
          * 
          * \returns The difference threshold */
        float
        getDifferenceThreshold () const
        {
          return difference_threshold_;
        }
        
        /** \brief Sets the precision of the occlusion testing
        *  Lower values mean more testing - the value specifies the interval (in voxels) between occlusion checks on the ray to the camera 
        *  \param[in] occlusion_interval_arg Distance between checks (default is 0.5) in multiples of resolution_ */  
        void
        setOcclusionTestInterval (const float occlusion_interval_arg)
        {
          occlusion_test_interval_ = occlusion_interval_arg;
        } 
        
        /** \brief Returns the interval for occlusion testing
         * 
         * \returns The interval */
        float
        getOcclusionTestInterval () const
        {
          return occlusion_test_interval_;
        }
        
        /** \brief Sets the number of threads to use 
          * 
          * \param[in] num_thread_arg Number of threads */        
        void
        setNumberOfThreads (const int num_thread_arg)
        {
          threads_ = num_thread_arg;
        }
        
        /** \brief Returns the maximum number of threads being used
          * 
          * \returns max number of threads */
        int
        getNumberOfThreads () const
        {
          return threads_;
        }
        
        /** \brief Returns a cloud containing only *new* voxels
          * 
          * \returns Cloud same as input type of only new voxels */
        typename PointCloudT::Ptr
        getNewVoxelCloud ();
        
        /** \brief Default implementation of leaf differencing function
          * 
          * \param[in] leaf The leaf to check for difference */
        static float 
        SeqVoxelDataDiff (const LeafContainerT* leaf);
        
      protected:
        /** \brief Tests whether input leaf-key pair is occluded from the camera view point
          *
          * \param[in] leaf_key_pair Leaf-Key pair to test for
          * \param[in] camera_key Key which specifies the camera position 
          * \returns 0 if path to camera is free, otherwise distance to occluder (in # of voxels) */
        float
        testForOcclusion (const LeafKeyPairT &leaf_key_pair, const OctreeKey& camera_key) const;
        
        /** \brief Fills in the neighbors fields for new voxels
          * \param[in] leaf_key_arg Leaf/Key pair of the voxel to compute neighbors for */
        void
        computeNeighbors (LeafKeyPairT& leaf_key_arg);
        
        /** \brief Checks if specified leaf has new neighbors
          * \param[in] leaf_key_arg Leaf/Key pair of the voxel to check for new neighbors 
          * \returns True/false whether leaf_key_arg has a new neighbor */
        bool
        testForNewNeighbors (const LeafContainerT* leaf_container) const;
        
        /** \brief Adds a sequential point index from input_ to the tree, threadsafe
         * \param[in] point_index_arg index of the point from input_ to add */
        void
        addPointSequential (const int point_index_arg);
        
        //! Local leaf pointer & octree key vector used to make iterating through leaves fast  
        LeafKeyVectorT leaf_key_vec_;
        
      private:
        //Stores a pointer to a difference function
        boost::function<float (const LeafContainerT* leaf)> diff_func_;
        
        //Stores pairs of leaf pointers & octree key ptrs to leaves
        LeafKeyVectorT new_frame_pairs_;
        
        //Stores the maximum difference allowed between sequential voxels before it is called "changed"
        float difference_threshold_;
        
        //Stores the precision for occlusion testing
        float occlusion_test_interval_;
        
        //Stores the camera position which is used for occlusion testing
        OctreeKey camera_key_;
        
        //Mutex lock to prevent simultaneous leaf creation
        boost::mutex create_mutex_;
        
        //Maximum number of threads to use 
        int threads_;
        
        //Stores whether the keys stored in the key vector are currently valid
        bool stored_keys_valid_;
        
        //Private members from parent OctreePointCloud class
        using OctreePointCloudT::input_;
        using OctreePointCloudT::resolution_;
        using OctreePointCloudT::min_x_;
        using OctreePointCloudT::min_y_;
        using OctreePointCloudT::min_z_;
        using OctreePointCloudT::max_x_;
        using OctreePointCloudT::max_y_;
        using OctreePointCloudT::max_z_;
        using OctreeBaseT::octree_depth_;
        using OctreeBaseT::max_key_;
    };
  }  
  
}









//#ifdef PCL_NO_PRECOMPILE
#include <pcl/octree/impl/octree_pointcloud_sequential.hpp>
//#endif

#endif //PCL_OCTREE_POINTCLOUD_SEQUENTIAL_H_
