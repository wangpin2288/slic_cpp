#include "Slic.cpp"

int main(int argc,char* argv[])
{
   Mat img = imread(argv[1]);
//   namedWindow("Original Image",1);
//   imshow("Original Image",img);
   Slic slic(img,200);
   slic.Start();
   slic.DrawLine();
      
   return 0;
}
