PCL_SUBSYS_DEPEND(build ${SUBSYS_NAME} DEPS visualization tracking octree)

# Find VTK 
if(VTK_FOUND)
  include (${VTK_USE_FILE})
  PCL_ADD_EXAMPLE(pcl_example_sequential_octree FILES example_sequential_octree.cpp
                  LINK_WITH pcl_common pcl_octree pcl_visualization)
endif(VTK_FOUND)
