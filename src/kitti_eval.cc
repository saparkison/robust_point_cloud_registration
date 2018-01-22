#include <iostream>
#include <chrono>
#include <thread>
#include <cmath>
#include <string>
#include <dirent.h>
#include <algorithm>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>
#include <pcl/console/parse.h>
#include <sophus/se3.hpp>
#include <sophus/types.hpp>
#include <sophus/common.hpp>
#include "kitti_metrics.h"
#include "filter_range.h"
#include "read_confusion_matrix.h"

#include "robust_pcl_registration/pda.h"


std::vector<std::string>
get_pcd_in_dir(std::string dir_name) {

    DIR           *d;
    struct dirent *dir;
    d = opendir(dir_name.c_str());

    std::vector<std::string> pcd_fns;

    if (d) {
        while ((dir = readdir(d)) != NULL) {

            // Check to make sure this is a pcd file match
            if (std::strlen(dir->d_name) >= 4 &&
                    std::strcmp(dir->d_name + std::strlen(dir->d_name)  - 4, ".pcd") == 0) {
                pcd_fns.push_back(dir_name + "/" + std::string(dir->d_name));
            }
        }

        closedir(d);
    }

    return pcd_fns;
}

int
main (int argc, char** argv)
{
    std::string strDirectory;
    std::string strGTFile;
    std::string strCMFile;
    if ( !pcl::console::parse_argument(argc, argv, "-s", strDirectory) ) {
        std::cout << "Need source directory (-s)\n";
        return (-1);
    }
    if ( !pcl::console::parse_argument(argc, argv, "-t", strGTFile) ) {
        std::cout << "Need ground truth file (-t)\n";
        return (-1);
    }
    if ( !pcl::console::parse_argument(argc, argv, "-m", strCMFile) ) {
        std::cout << "Need ground confusion matrix file (-m)\n";
        return (-1);
    }
    Eigen::Matrix<double, 11, 11> cm = ReadConfusionMatrix<11>(strCMFile);
    std::cout << "Confusion Matrix:\n" << cm << std::endl;

    std::vector<std::string> pcd_fns = get_pcd_in_dir(strDirectory);
    std::sort(pcd_fns.begin(),pcd_fns.end());

    std::cout << "PCD FILES\n";
    for(std::string s: pcd_fns) {
        std::cout << s << std::endl;
    }

    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%d-%m-%Y,%H-%M-%S");
    auto dateStr = oss.str();

/*
    std::ofstream foutSICP;
    //foutSICP.open(dateStr+"SICPkitti.csv");

    std::ofstream foutGICP;
    //foutGICP.open(dateStr+"GICPkitti.csv");

    std::ofstream foutse3GICP;
    //foutse3GICP.open(dateStr+"se3GICPkitti.csv");

    KittiMetrics semanticICPMetrics(strGTFile);
    KittiMetrics se3GICPMetrics(strGTFile);
    KittiMetrics GICPMetrics(strGTFile);
  */  

    std::ofstream foutRICP;
    foutRICP.open(dateStr+"RICPkitti.csv");
    KittiMetrics RICPMetrics(strGTFile, &foutRICP);
    std::ofstream foutBootstrap;
    foutBootstrap.open(dateStr+"RICPkitti.csv");
    KittiMetrics bootstrapMetrics(strGTFile, &foutBootstrap);

    IpdaParameters ipda_params;
    ipda_params.save_aligned_cloud = false;
    ipda_params.solver_minimizer_progress_to_stdout = true;
    ipda_params.solver_use_nonmonotonic_steps = true;
    ipda_params.use_gaussian = true;
    ipda_params.visualize_clouds = false;
    ipda_params.dof = 100.0;
    ipda_params.point_size_aligned_source = 3.0;
    ipda_params.point_size_source = 3.0;
    ipda_params.point_size_target = 3.0;
    ipda_params.radius = 1.5;
    ipda_params.solver_function_tolerance = 10e-16;
    ipda_params.source_filter_size = 5.0;
    ipda_params.target_filter_size = 0.0;
    ipda_params.transformation_epsilon = 1e-3;
    ipda_params.dimension = 3;
    ipda_params.maximum_iterations = 50;
    ipda_params.max_neighbours = 8;
    ipda_params.solver_maximum_iterations = 100;
    ipda_params.solver_num_threads = 8;
    ipda_params.aligned_cloud_filename = "alinged.pcd";
    ipda_params.frame_id = "map";
    ipda_params.source_cloud_filename = "in.pcd";
    ipda_params.target_cloud_filename = "out.pcd";
    for(size_t n = 0; n<(pcd_fns.size()-3); n+=3) {
        std::cout << "Cloud# " << n << std::endl;
        size_t indxTarget = n;
        size_t indxSource = indxTarget + 3;
        std::string strTarget = pcd_fns[indxTarget];
        std::string strSource = pcd_fns[indxSource];
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudA (new pcl::PointCloud<pcl::PointXYZ>);

        if (pcl::io::loadPCDFile<pcl::PointXYZ> (strSource, *cloudA) == -1) //* load the file
        {
            PCL_ERROR ("Couldn't read source file\n");
            return (-1);
        }

        filterRange(cloudA, 40.0);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudB (new pcl::PointCloud<pcl::PointXYZ>);

        if (pcl::io::loadPCDFile<pcl::PointXYZ> (strTarget, *cloudB) == -1) //* load the file
        {
            PCL_ERROR ("Couldn't read target file\n");
            return (-1);
        }

        filterRange(cloudB, 40.0);
        auto begin = std::chrono::steady_clock::now();
        //Sophus::SE3d initTransform = semanticICPMetrics.getGTtransfrom(n, n+3);
        Eigen::Matrix4d temp = Eigen::Matrix4d::Identity();
        //Bootstrap boot(cloudA, cloudB);
        //Eigen::Matrix4d temp = (boot.align()).cast<double>();
        Sophus::SE3d initTransform(temp);
        auto end = std::chrono::steady_clock::now();
        int timeInit = std::chrono::duration_cast<std::chrono::seconds>(end-begin).count();
        std::cout << "Init MSE "
                  << bootstrapMetrics.evaluate(initTransform, indxTarget, indxSource, timeInit)
                  << std::endl;

        // Run IPDA.
        Ipda ipda(ipda_params);
        pcl::PointCloud<pcl::PointXYZ>::Ptr
          finalCloudem( new pcl::PointCloud<pcl::PointXYZ> );

        begin = std::chrono::steady_clock::now();
        Eigen::Affine3d t= ipda.evaluate(cloudA, cloudB);
        end = std::chrono::steady_clock::now();
        int timeRICP = std::chrono::duration_cast<std::chrono::seconds>(end-begin).count();
        std::cout << "Time RICP: "
                << timeRICP << std::endl;
        Sophus::SE3d ricpTranform = Sophus::SE3d::fitToSE3(t.matrix());
        std::cout << "RICP MSE: "
                  << RICPMetrics.evaluate(ricpTranform, indxTarget, indxSource, timeRICP)
                  << std::endl;
    }
    std::cout << " RICP FINAL MSE: " << RICPMetrics.getTransformMSE() << std::endl;
    std::cout << "Transform\n";
    RICPMetrics.printTransfrom();
    std::cout << "Rot\n";
    RICPMetrics.printRot();
    std::cout << "Trans\n";
    RICPMetrics.printTrans();
    foutRICP.close();

    return (0);
}
