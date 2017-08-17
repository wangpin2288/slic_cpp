#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <math.h>
#include "Slic.h"
#include "Pixel.hpp"
#include <iostream>

using namespace std;
using namespace cv;


Slic::Slic(Mat src,int num)
{
   Img = src;
   m_iWidth = Img.cols;  
   m_iHeight = Img.rows;
   m_iKernelNum=num;
   m_iS = sqrt(m_iWidth*m_iHeight/m_iKernelNum);
   m_iM = 10;

//初始化像素数组
   for(int i=0;i<m_iHeight;i++)
   {
      for(int j=0;j<m_iWidth;j++)
      {
         Pixel pixel(i,j,0,0,0);
         PixelArray.push_back(pixel);   
      } 
   }

//初始化聚类中心坐标
   for(int i=0;i<m_iKernelNum;i++)
   {
      KernelArray.push_back(0);
   }

}



float Slic::Gama(float x)//非线性色调编辑函数
{
   if(x>0.04045) return pow((x+0.055f)/1.055f,2.4f);
   else return x/12.92f;
}

/*
float Slic::F(float x)
{
   if(x>0.008856f) return pow(x,1.0f/3.0f);
   else return 7.787f*x+0.137931f; 
}
*/


void Slic::RGB2LAB()//颜色空间转换
{
   for(int i=0;i<m_iHeight;i++)
   {
      for(int j=0;j<m_iWidth;j++)
      {
          float B = Gama((float)Img.at<Vec3b>(i,j)[0]/255.0);   
          float G = Gama((float)Img.at<Vec3b>(i,j)[1]/255.0);  
          float R = Gama((float)Img.at<Vec3b>(i,j)[2]/255.0);
          
          float X = R*0.4124+G*0.3576+B*0.1805;
          float Y = R*0.2126+G*0.7152+B*0.0722;
          float Z = R*0.0193+G*0.1192+B*0.9505;

          X/=0.95047;
          Y/=1.0;
          Z/=1.08883;

          float FX = X > 0.008856f ? pow(X,1.0f/3.0f) : (7.787f*X+0.137931f);
          float FY = Y > 0.008856f ? pow(Y,1.0f/3.0f) : (7.787f*X+0.137931f);
          float FZ = Z > 0.008856f ? pow(Z,1.0f/3.0f) : (7.787f*Z+0.137931f);

          float l = Y > 0.008856f ? (116.0f * FY - 16.0F) : (903.3f*Y);
          float a = 500.0f*(FX-FY);
          float b = 200.0f*(FY-FZ);

          PixelArray[i*m_iWidth+j].m_fL = l;
          PixelArray[i*m_iWidth+j].m_fA = a;
          PixelArray[i*m_iWidth+j].m_fB = b;
      } 
   }

   cout<<"RGB2LAB..."<<endl;
}

void Slic::InitKernel()//初始化聚类中心
{
   cout<<"Start Initkernel..."<<endl;
   RGB2LAB();
//确定种子点的坐标
   int count = 0;
   for(int i=m_iS/2;i<m_iHeight-m_iS/2;i+=m_iS)
   {
      for(int j=m_iS/2;j<m_iWidth-m_iS/2;j+=m_iS)
      {
         if(count==m_iKernelNum) break;
         KernelArray[count] = i*m_iWidth+j;
         PixelArray[i*m_iWidth+j].m_bIsKernel = 1;
         PixelArray[i*m_iWidth+j].m_iKernel = count;
         count++;
      }
   }

   for(int i=0;i<m_iKernelNum;i++)//遍历每一个聚类中心
   {
      int kernel_x = KernelArray[i]/m_iWidth;
      int kernel_y = KernelArray[i]%m_iWidth;
      for(int k = kernel_x-m_iS; k < kernel_x+m_iS; k++)
      {
        for(int l = kernel_y-m_iS; l < kernel_y+m_iS; l++)
        {
           if(k>=0 && k<m_iHeight && l>=0 && l<m_iWidth)
           {
              PixelArray[k*m_iWidth+l].m_iKernel = i;
              PixelArray[k*m_iWidth+l].Dist = Distance(k,l,kernel_x,kernel_y);
           }
        }
      }
   }


   cout<<"End InitKernel..."<<endl;
}


float Slic::Grad(int x,int y)//计算点(x,y)处的梯度值
{
   float dxl = pow((PixelArray[(x-1)*m_iWidth+y].m_fL-PixelArray[(x+1)*m_iWidth+y].m_fL),2);
   float dxa = pow((PixelArray[(x-1)*m_iWidth+y].m_fA-PixelArray[(x+1)*m_iWidth+y].m_fA),2);
   float dxb = pow((PixelArray[(x-1)*m_iWidth+y].m_fB-PixelArray[(x+1)*m_iWidth+y].m_fB),2);

   float dyl = pow((PixelArray[x*m_iWidth+y-1].m_fL-PixelArray[x*m_iWidth+y+1].m_fL),2);
   float dya = pow((PixelArray[x*m_iWidth+y-1].m_fA-PixelArray[x*m_iWidth+y+1].m_fA),2);
   float dyb = pow((PixelArray[x*m_iWidth+y-1].m_fB-PixelArray[x*m_iWidth+y+1].m_fB),2);
   return dxl+dxa+dxb+dyl+dya+dyb;
}


float Slic::Distance(int x1,int y1,int x2,int y2)//距离度量
{
   float l1=PixelArray[x1*m_iWidth+y1].m_fL;
   float a1=PixelArray[x1*m_iWidth+y1].m_fA;
   float b1=PixelArray[x1*m_iWidth+y1].m_fB;

   float l2=PixelArray[x2*m_iWidth+y2].m_fL;
   float a2=PixelArray[x2*m_iWidth+y2].m_fA;
   float b2=PixelArray[x2*m_iWidth+y2].m_fB;

   float dc=sqrt(pow(l1-l2,2)+pow(a1-a2,2)+pow(b1-b2,2));
   float ds=sqrt(pow(x1-x2,2)+pow(y1-y2,2));
   return sqrt(pow(dc/10.0,2)+pow(ds/m_iS,2));
}


void Slic::UpdateKernel(int n=3)//更新种子点
{
//在种子点kernel周围n*n邻域内计算重新选取种子点
   cout<<"Start UpdateKernel..."<<endl;
      for(int i=0;i<m_iKernelNum;i++)//遍历每一个种子点
      {
          int x=KernelArray[i]/m_iWidth; 
          int y=KernelArray[i]%m_iWidth;  
          int newx=x,newy=y;  
          for(int k=x-(n-1)/2;k<=x+(n-1)/2;k++)
          {
             for(int l=y-(n-1)/2;l<y+(n-1)/2;l++)
             {
                if(k>0 && k<m_iHeight-1 && l>0 && l<m_iWidth-1)
                {
                    if(Grad(k,l)<Grad(newx,newy)) 
                    {
                       newx=k;
                       newy=l;
                    } 
                }
                
            }

          }
          PixelArray[KernelArray[i]].m_bIsKernel = 0;
          KernelArray[i]=newx*m_iWidth+newy;
          PixelArray[KernelArray[i]].m_bIsKernel = 1;
      }


   //为每个像素周围领域内分配类别标签,遍历每个种子点，对它周围2S＊2S内的像素核距离进行计算
   for(int i=0;i<m_iKernelNum;i++)//遍历每一个聚类中心
   {
      int kernel_x = KernelArray[i]/m_iWidth;
      int kernel_y = KernelArray[i]%m_iWidth;
      for(int k = kernel_x-m_iS; k < kernel_x+m_iS; k++)
      {
        for(int l = kernel_y-m_iS; l < kernel_y+m_iS; l++)
        {
           if(k>=0 && k<m_iHeight && l>=0 && l<m_iWidth &&  PixelArray[k*m_iWidth+l].Dist > Distance(k,l,kernel_x,kernel_y) )
           {
              PixelArray[k*m_iWidth+l].m_iKernel = i;
              PixelArray[k*m_iWidth+l].Dist = Distance(k,l,kernel_x,kernel_y);
           }
        }
      }
   }




/*
遍历每一个像素，对该像素周围2S＊2S范围类的种子点进行计算，重新选取聚类中心
   for(int i=0;i<m_iHeight;i++)
   {
      for(int j=0;j<m_iWidth;j++)
      {
         float minDist=PixelArray[i*m_iWidth+j].Dist;
         for(int k=i-m_iS;k<i+m_iS;k++)
           for(int l=j-m_iS;l<j+m_iS;l++)
           {
              if(k>=0 && k<m_iHeight && l>=0 && l<m_iWidth &&
                 PixelArray[k*m_iWidth+l].m_bIsKernel && Distance(i,j,k,l)<minDist)
              {
                 minDist = Distance(i,j,k,l);
                 PixelArray[i*m_iWidth+j].m_iKernel = PixelArray[k*m_iWidth+l].m_iKernel;
              }

           }

      }

   }
*/
    cout<<"End Update Kernel..."<<endl;
}


void Slic::Start()
{
   int n=20;
   InitKernel();
   while(n>0)
   {
      UpdateKernel();
      n--;
      cout<<"Update Kernel... "<<n<<endl;
   }
}

void Slic::DrawLine()
{


    for(int i=1;i<m_iHeight-1;i++)
    {
       for(int j=1;j<m_iWidth-1;j++)
       {

           int ker = PixelArray[i*m_iWidth+j].m_iKernel;
           int ker_up = PixelArray[(i-1)*m_iWidth+j].m_iKernel;
     //     int ker_down = PixelArray[(i+1)*m_iWidth+j].m_iKernel;
           int ker_left = PixelArray[i*m_iWidth+j-1].m_iKernel;
      //     int ker_right = PixelArray[i*m_iWidth+j+1].m_iKernel;

           if(ker!=ker_up || ker!=ker_left)
           {
               Img.at<Vec3b>(i,j)[0]=0;
               Img.at<Vec3b>(i,j)[1]=0;
               Img.at<Vec3b>(i,j)[2]=0;
          
           }

       }
    }

   for(int i=0;i<m_iKernelNum;i++)
   {
      int k_x = KernelArray[i]/m_iWidth;
      int k_y = KernelArray[i]%m_iWidth;
      Img.at<Vec3b>(k_x,k_y)[0]=0;
      Img.at<Vec3b>(k_x,k_y)[1]=0;
      Img.at<Vec3b>(k_x,k_y)[2]=255;
   }


   namedWindow("SLIC Output",1);
   imshow("SLIC Output",Img);
   imwrite("./slic_output.jpg",Img);
   waitKey(0);

}


















