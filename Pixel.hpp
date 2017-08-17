#ifndef PIXEL_HPP
#define PIXEL_HPP

#include <stdlib.h>
using namespace std;


class Pixel
{
public:
   Pixel(int x,int y,int l,int a,int b);
   ~Pixel(){};
   int m_iX;
   int m_iY;
   double m_fL;
   double m_fA;   
   double m_fB;
  
   double Dist;//该像素点距离其聚类中心的距离
   bool m_bIsKernel;
   int m_iKernel;
};

Pixel::Pixel(int x,int y,int l,int a,int b)
{
   m_iX=x;
   m_iY=y;
   m_fL=l;
   m_fA=a;
   m_fB=b;
   m_bIsKernel=0;
   m_iKernel=0;
}

#endif
