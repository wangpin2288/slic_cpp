#ifndef SLIC_H
#define SLIC_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdlib.h>
#include <vector>
#include <math.h>
#include "Pixel.hpp"
using namespace cv;
using namespace std;

class Slic
{
public:
   Slic(Mat src,int num);
   ~Slic(){};
   float Gama(float x);//非线性色调编辑函数
   float F(float x);
   void RGB2LAB();//颜色空间转换
   void InitKernel();//初始化聚类中心
   float Grad(int x,int y);//计算点(x,y)处的梯度值
   float Distance(int x1,int y1,int x2,int y2);//距离度量
   void UpdateKernel(int n);//在种子点kernel周围n*n邻域内计算重新选取种子点
   void Start();
   void DrawLine();

private:
   int m_iWidth;
   int m_iHeight;
   int m_iKernelNum;  //初始化种子点数目
   vector<int> KernelArray;//聚类中心数组，存储聚类中心的坐标值
   vector<Pixel> PixelArray;//所有像素点数组 

   int m_iS;  //最大空间距离
   int m_iM;  //最大颜色距离  
   Mat Img;
};


#endif

