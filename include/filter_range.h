#ifndef _FILTER_RANGE_H_
#define _FILTER_RANGE_H_



void filterRange(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPtr,
                 const double range) {

    auto it = cloudPtr->begin();
    while( it != cloudPtr->end()) {
        pcl::PointXYZ pt = *it;
        if((pt.x*pt.x+pt.y*pt.y+pt.z*pt.z)>range*range) {
            cloudPtr->erase(it);
        } else {
            it++;
        }
    }
}

#endif // #ifndef _FILTER_RANGE_H_
