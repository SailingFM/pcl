/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2014-, Open Perception, Inc.
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
 *   * Neither the name of the copyright holder(s) nor the names of its
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
 */

#ifndef PCL_SEGMENTATION_LCCP_SEGMENTATION_H_
#define PCL_SEGMENTATION_LCCP_SEGMENTATION_H_

#include <pcl/pcl_base.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/segmentation/supervoxel_clustering.h>

#define PCL_INSTANTIATE_LCCPSegmentation(T) template class PCL_EXPORTS pcl::LCCPSegmentation<T>;

namespace pcl
{
  /** \brief A simple segmentation algorithm partitioning a supervoxel graph into groups of locally convex connected supervoxels separated by concave borders.
   *  \note If you use this in a scientific work please cite the following paper:
   *  S. C. Stein, M. Schoeler, J. Papon, F. Woergoetter
   *  Object Partitioning using Local Convexity
   *  In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2014
   *  \author Simon Christoph Stein and Markus Schoeler (mschoeler@gwdg.de)
   *  \ingroup segmentation
   */
  template <typename PointT>
  class LCCPSegmentation
  {
    /** \brief Edge Properties stored in the adjacency graph.*/
    struct EdgeProperties
    {
      bool is_convex;
      EdgeProperties () :
      is_convex (false)
      {
      }
    };

    public:

      // Adjacency list with nodes holding labels (uint32_t) and edges holding EdgeProperties.
      typedef typename boost::adjacency_list<boost::setS, boost::setS, boost::undirectedS, uint32_t, EdgeProperties> SupervoxelAdjacencyList;
      typedef typename boost::graph_traits<SupervoxelAdjacencyList>::vertex_iterator VertexIterator;
      typedef typename boost::graph_traits<SupervoxelAdjacencyList>::adjacency_iterator AdjacencyIterator;

      typedef typename boost::graph_traits<SupervoxelAdjacencyList>::vertex_descriptor VertexID;
      typedef typename boost::graph_traits<SupervoxelAdjacencyList>::edge_iterator EdgeIterator;
      typedef typename boost::graph_traits<SupervoxelAdjacencyList>::out_edge_iterator OutEdgeIterator;
      typedef typename boost::graph_traits<SupervoxelAdjacencyList>::edge_descriptor EdgeID;

      LCCPSegmentation ();
      virtual
      ~LCCPSegmentation ();

      /** \brief Reset internal memory.  */
      void
      reset ();

      /** \brief Merge supervoxels using local convexity. The input parameters are generated by using the SupervoxelClustering class.
       *  \param[in] supervoxel_clusters_arg Map of < supervoxel labels, supervoxels >
       *  \param[in] label_adjacency_arg The graph defining the supervoxel adjacency relations  */
      void
      segment (std::map<uint32_t, typename pcl::Supervoxel<PointT>::Ptr> &supervoxel_clusters_arg,
               std::multimap<uint32_t, uint32_t> &label_adjacency_arg);

      /** \brief Relabels cloud with supervoxel labels with the computed segment labels
       *  \param[in] labeled_cloud_arg Cloud to relabel  */
      void
      relabelCloud (pcl::PointCloud<pcl::PointXYZL>::Ptr labeled_cloud_arg);
      
      /** \brief Segments smaller than segment_size are assigned to label of largest neighbor.
       *  \param[in] min_segment_size_arg Segments smaller than this size will be filtered
       *  \note Currently this runs multiple times, until no segment < min_segment_size is found. Could be faster.  */
      void
      removeSmallSegments (uint32_t min_segment_size_arg);

      /** \brief Get map<SegmentID, std::vector<SuperVoxel IDs> >
       *  \param[out] segment_supervoxel_map_arg The output container. On error the map is empty. */
      inline void
      getSegmentSupervoxelMap (std::map<uint32_t, std::vector<uint32_t> >& segment_supervoxel_map_arg) const
      {
        if (grouping_data_valid_)
        {
          segment_supervoxel_map_arg = seg_label_to_sv_list_map_;
        }
        else
        {
          PCL_WARN ("[pcl::LCCPSegmentation::getSegmentMap] WARNING: Call function segment first. Nothing has been done. \n");
          segment_supervoxel_map_arg = std::map<boost::uint32_t, std::vector<boost::uint32_t> > ();
        }
      }
      
      /** \brief Get map <SegmentID, std::set<Neighboring SegmentIDs> >
       * \param[out] segment_adjacency_map_arg map < SegmentID, std::set< Neighboring SegmentIDs> >. On error the map is empty.  */
      inline void
      getSegmentAdjacencyMap (std::map<uint32_t, std::set<uint32_t> >& segment_adjacency_map_arg)
      {
        if (grouping_data_valid_)
        {
          if (seg_label_to_neighbor_set_map_.empty ())
            computeSegmentAdjacency ();
          segment_adjacency_map_arg = seg_label_to_neighbor_set_map_;
        }
        else
        {
          PCL_WARN ("[pcl::LCCPSegmentation::getSegmentAdjacencyMap] WARNING: Call function segment first. Nothing has been done. \n");
          segment_adjacency_map_arg = std::map<boost::uint32_t, std::set<boost::uint32_t> > ();
        }
      }
      
      /** \brief Get normal threshold
       *  \return The concavity tolerance angle in [deg] that is currently set */
      inline float
      getConcavityToleranceThreshold () const
      {
        return (concavity_tolerance_threshold_);
      }
      
      /** \brief Get the supervoxel adjacency graph with classified edges (boost::adjacency_list).
       * \param[out] adjacency_list_arg The supervoxel adjacency list with classified (convex/concave) edges. On error the list is empty.  */
      inline void
      getSVAdjacencyList (SupervoxelAdjacencyList& adjacency_list_arg) const
      {
        if (grouping_data_valid_)
        {
          adjacency_list_arg = sv_adjacency_list_;
        }
        else
        {
          PCL_WARN ("[pcl::LCCPSegmentation::getSVAdjacencyList] WARNING: Call function segment first. Nothing has been done. \n");
          adjacency_list_arg = pcl::LCCPSegmentation<PointT>::SupervoxelAdjacencyList ();
        }
      }
      
      /** \brief Set normal threshold
       *  \param[in] concavity_tolerance_threshold_arg the concavity tolerance angle in [deg] to set */
      inline void
      setConcavityToleranceThreshold (float concavity_tolerance_threshold_arg)
      {
        concavity_tolerance_threshold_ = concavity_tolerance_threshold_arg;
      }

      /** \brief Determines if a smoothness check is done during segmentation, trying to invalidate edges of non-smooth connected edges (steps). Two supervoxels are unsmooth if their plane-to-plane distance DIST > (expected_distance + smoothness_threshold_*voxel_resolution_). For parallel supervoxels, the expected_distance is zero.
       *  \param[in] use_smoothness_check_arg Determines if the smoothness check is used
       *  \param[in] voxel_res_arg The voxel resolution used for the supervoxels that are segmented
       *  \param[in] seed_res_arg The seed resolution used for the supervoxels that are segmented
       *  \param[in] smoothness_threshold_arg Threshold (/fudging factor) for smoothness constraint according to the above formula. */
      inline void
      setSmoothnessCheck (bool use_smoothness_check_arg,
                          float voxel_res_arg,
                          float seed_res_arg,
                          float smoothness_threshold_arg = 0.1)
      {
        use_smoothness_check_ = use_smoothness_check_arg;
        voxel_resolution_ = voxel_res_arg;
        seed_resolution_ = seed_res_arg;
        smoothness_threshold_ = smoothness_threshold_arg;
      }

      /** \brief Determines if we want to use the sanity criterion to invalidate singular connected patches
       *  \param[in] use_sanity_criterion_arg Determines if the sanity check is performed */
      inline void
      setSanityCheck (bool use_sanity_criterion_arg)
      {
        use_sanity_check_ = use_sanity_criterion_arg;
      }

      /** \brief Set the value used for k convexity. For k>0 convex connections between p_i and p_j require k common neighbors of these patches that have a convex connection to both.
       *  \param[in] k factor used for extended convexity check */
      inline void
      setKFactor (uint k)
      {
        k_factor_ = k;
      }

    private:

      /** \brief Compute the adjacency of the segments */
      void
      computeSegmentAdjacency ();

      /** \brief Is called within groupSupervoxels mainly to reserve required memory.
       *  \param[in] supervoxel_clusters_arg map of < supervoxel labels, supervoxels >
       *  \param[in] label_adjacency_arg The graph defining the supervoxel adjacency relations */
      void
      prepareSegmentation (const std::map<uint32_t, typename pcl::Supervoxel<PointT>::Ptr> &supervoxel_clusters_arg,
                           const std::multimap<uint32_t, uint32_t> &label_adjacency_arg);

      /** \brief Assigns neighbors of the query point to the same group as the query point. Recursive part of groupSupervoxels (..). Grouping is done by a depth-search of nodes in the adjacency-graph.
       *  \param[in] queryPointID ID of point whose neighbors will be considered for grouping
       *  \param[in] group_label ID of the group/segment the queried point belongs to  */
      void
      recursiveGrouping (VertexID const &queryPointID,
                         unsigned int const group_label);

      /** \brief Calculates convexity of edges and saves this to the adjacency graph.
       *  \param[in] adjacency_list_arg The supervoxel adjacency list*/
      void
      calculateConvexConnections (SupervoxelAdjacencyList& adjacency_list_arg);

      /** \brief Connections are only convex if this is true for at least k common neighbors of the two patches. Call setKFactor (..) before segment (..) to use this.
       *  \param[in] k_arg Factor used for extended convexity check */
      void
      applyKconvexity (uint k_arg);

      /** \brief Returns true if the connection between source and target is convex.
       *  \param[in] source_label_arg Label of one Supervoxel connected to the edge that should be checked
       *  \param[in] target_label_arg Label of the other Supervoxel connected to the edge that should be checked
       *  \return True if connection is convex */
      bool
      connIsConvex (uint32_t source_label_arg,
                    uint32_t target_label_arg);

      ///  *** Parameters *** ///

      /** \brief Normal Threshold in degrees [0,180] used for merging */
      float concavity_tolerance_threshold_;

      /** \brief Marks if valid grouping data (sv_adjacency_list_, svLabel_segLabel_map_, processed_) is avaiable */
      bool grouping_data_valid_;

      /** \brief Determines if the smoothness check is used during segmentation*/
      bool use_smoothness_check_;

      /** \brief Two supervoxels are unsmooth if their plane-to-plane distance DIST >  (expected_distance + smoothness_threshold_*voxel_resolution_). For parallel supervoxels, the expected_distance is zero. */
      float smoothness_threshold_;

      /** \brief Determines if we use the sanity check which tries to find and invalidate singular connected patches*/
      bool use_sanity_check_;
      
      /** \brief Seed resolution of the supervoxels (used only for smoothness check) */
      float seed_resolution_;

      /** \brief Voxel resolution used to build the supervoxels (used only for smoothness check)*/
      float voxel_resolution_;

      /** \brief Factor used for k-convexity */
      uint k_factor_;

      /** \brief Stores which SuperVoxel labels were already visited during recursive grouping.    processed_[sv_Label] = false (default)/true (already processed) */
      std::map<uint32_t, bool> processed_;

      /** \brief Adjacency graph with the supervoxel labels as nodes and edges between adjacent supervoxels */
      SupervoxelAdjacencyList sv_adjacency_list_;

      /** \brief map from the supervoxel labels to the supervoxel objects  */
      std::map<uint32_t, typename pcl::Supervoxel<PointT>::Ptr> sv_label_to_supervoxel_map_;

      /** \brief Storing relation between original SuperVoxel Labels and new segmantion labels. svLabel_segLabel_map_[old_labelID] = new_labelID */
      std::map<uint32_t, uint32_t> sv_label_to_seg_label_map_;

      /** \brief map < Segment Label, std::vector< SuperVoxel Labels> > */
      std::map<uint32_t, std::vector<uint32_t> > seg_label_to_sv_list_map_;

      /** \brief map < SegmentID, std::vector< Neighboring segment labels> > */
      std::map<uint32_t, std::set<uint32_t> > seg_label_to_neighbor_set_map_;

  };
}

#ifdef PCL_NO_PRECOMPILE
#include <pcl/segmentation/impl/lccp_segmentation.hpp>
#endif

#endif // PCL_SEGMENTATION_LCCP_SEGMENTATION_H_
