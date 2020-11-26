#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <time.h>
#include <math.h>

using namespace cv;
using namespace std;

//讀取raw檔之副程式以方便之後撰寫程式
void Read_Raw(string file_path,unsigned char* output,int height,int width)
{
  char* temp = &file_path[0] ;
  FILE *inputFile;
  inputFile = fopen(temp,"rb");
  fread(output,1,height * width,inputFile);
  fclose(inputFile);
}
//寫入raw檔之副程式以方便之後撰寫程式
void Write_Raw(string file_path,unsigned char* input,int height,int width)
{
  char* temp = &file_path[0] ;
  FILE *outputFile;
  outputFile = fopen(temp,"w");
  fwrite(input,1,height * width,outputFile);
  fclose(outputFile);
}
float Interpolate(int a0,int a1,int a2,int a3,float x) //bicubic 副程式
{
  float p1,p2,p3,p4,ps;
  p1=(float)(a3-(2*a2)+(2*a1)-a0)/6;
  p2=(float)(a0-a1)/2;
  p3=(float)(a3-(8*a2)+(5*a1)+(2*a0))/(-6);
  p4=(float)(a1);
  ps=(p1*x*x*x)+(p2*x*x)+(p3*x)+p4;
  return ps;
}
void Bicubic(Mat inputMat,Mat outputMat)
{
  float shrinkx,shrinky;
  shrinkx=(float)outputMat.rows/inputMat.rows;
  shrinky=(float)outputMat.cols/inputMat.cols;
  float x,y;
  int pta[4],ptx[4],pty[4];
  float pt[4];
  int pxy;
  for(int i=0;i<outputMat.cols;i++)
  {
    y=i/shrinky;
    if(y<1)
    {
      for(int k=0;k<4;k++)
        pty[k]=k;
    }
    else if(y>(inputMat.cols-2))
    {
      pty[3]=inputMat.cols-1; 
      pty[2]=pty[3]-1;
      pty[1]=pty[2]-1;
      pty[0]=pty[1]-1;
    } 
    else
    {
      pty[0]=floor(y)-1;
      pty[1]=floor(y);
      pty[2]=floor(y)+1;
      pty[3]=floor(y)+2;
    }
    y=y-pty[0];
    for(int j=0;j<outputMat.rows;j++)
    {
      x=j/shrinkx;
      if(x<1)
      {
        for(int k=0;k<4;k++)
          ptx[k]=k;
      }
      else if(x>(inputMat.rows-2))
      {
        ptx[3]=inputMat.rows-1; 
        ptx[2]=ptx[3]-1;
        ptx[1]=ptx[2]-1;
        ptx[0]=ptx[1]-1;
      } 
      else
      {
        ptx[0]=floor(x)-1;
        ptx[1]=floor(x);
        ptx[2]=floor(x)+1;
        ptx[3]=floor(x)+2;
      }
      x=x-ptx[0];
      for(int c=0;c<4;c++)
      {
        for(int r=0;r<4;r++)
          pta[r]=inputMat.at<uchar>(ptx[r],pty[c]);
        pt[c]=Interpolate(pta[0],pta[1],pta[2],pta[3],x);
      }
      pxy=Interpolate(pt[0],pt[1],pt[2],pt[3],y);
      outputMat.at<uchar>(j,i)=pxy;
    }
  }
}
void Gray_level_resolution(Mat inputMat,Mat outputMat,int num)
{
  unsigned char filter=0;
  for(int i=0;i<8-num;i++)
  {
    filter=(filter*2+1);
  }
  filter=~filter;
  for(int i=0;i<inputMat.cols;i++)
  {
    for(int j=0;j<inputMat.rows;j++)
      outputMat.at<uchar>(j,i)=inputMat.at<uchar>(j,i) & filter;
  }
}
float MSE(Mat inputMat,Mat outputMat)
{
  double sum=0;
  int count=0,a;
  float mse=0;
  for(int i=0;i<inputMat.cols;i++)
  {
    for(int j=0;j<inputMat.rows;j++)
    {
      a=outputMat.at<uchar>(j,i)-inputMat.at<uchar>(j,i);
      count++;
      sum=sum+(a*a);
    }
  }
  mse=(float)sum/count;
  return mse;
}
float PSNR(Mat inputMat,Mat outputMat)
{
  double sum=0;
  int count=0,a;
  float mse=0,psnr=0;
  for(int i=0;i<inputMat.cols;i++)
  {
    for(int j=0;j<inputMat.rows;j++)
    {
      a=outputMat.at<uchar>(j,i)-inputMat.at<uchar>(j,i);
      count++;
      sum=sum+(a*a);
    }
  }
  mse=(float)sum/count;

  psnr=10*log10((255*255)/mse);
  return psnr;
}
void Delay(int number_of_seconds) 
{ 
    // Converting time into milli_seconds 
    int milli_seconds =  number_of_seconds; 
  
    // Storing start time 
    clock_t start_time = clock(); 
  
    // looping till required time is not achieved 
    while (clock() < start_time + milli_seconds) 
        ; 
} 
void Difference(Mat inputMat,Mat outputMat)
{
  int length = inputMat.cols;
  int width = inputMat.rows;
  int max=-255,min=255,data;
  for(int i=0;i<length;i++)
  {
    for(int j=0;j<width;j++)
    {
      if(j==0)
      {
        outputMat.at<uchar>(i,j)=inputMat.at<uchar>(i,j);
        
      }
      else
      {
        data=inputMat.at<uchar>(i,j)-inputMat.at<uchar>(i,j-1);
        outputMat.at<uchar>(i,j)=data;
        if(data<min)
          min=data;
        else if(data>max)
          max=data;
      }
    }
  }
  printf("max=%d,min=%d",max,min);
}
void Undifference(Mat inputMat,Mat outputMat)
{
  int length = inputMat.cols;
  int width = inputMat.rows;

  for(int i=0;i<length;i++)
  {
    for(int j=0;j<width;j++)
    {
      if(j==0)
      {
        outputMat.at<uchar>(i,j)=inputMat.at<uchar>(i,j);
      }
      else
      {
        outputMat.at<uchar>(i,j)=outputMat.at<uchar>(i,j-1)+inputMat.at<uchar>(i,j);
      }
    }
  }
}
void MergeImage(Mat inputMat1,Mat inputMat2,Mat outputMat,int mainbit)
{
  int length = inputMat1.cols;
  int width = inputMat1.rows;
  int data=0;
  int subbit=8-mainbit;
  Mat pic1(length,width,CV_8UC1);
  Mat pic2(length,width,CV_8UC1);
  Gray_level_resolution(inputMat1,pic1,mainbit);
  Gray_level_resolution(inputMat2,pic2,subbit);

  for(int i=0;i<length;i++)
  {
    for(int j=0;j<width;j++)
    {
      data=pic2.at<uchar>(i,j);
      data=data>>mainbit;
      outputMat.at<uchar>(i,j)=pic1.at<uchar>(i,j)+data;
    }
  }
}
void SplitImage(Mat inputMat,Mat outputMat1,Mat outputMat2,int mainbit)

{
  int length = inputMat.cols;
  int width = inputMat.rows;
  int data=0;
  Mat pic1(length,width,CV_8UC1);
  Gray_level_resolution(inputMat,pic1,mainbit);
  for(int i=0;i<length;i++)
  {
    for(int j=0;j<width;j++)
    {
      outputMat1.at<uchar>(i,j)=pic1.at<uchar>(i,j);
      data=inputMat.at<uchar>(i,j)-pic1.at<uchar>(i,j);
      outputMat2.at<uchar>(i,j)=data<<mainbit;
    }
  }
}

void mixture(Mat Image,Mat pica,Mat picb,Mat picc,Mat picd,Mat pice,Mat picf,Mat picg,Mat pich)//MSB>>LSB
{
  int length = Image.cols;
  int width = Image.rows;
  int data=0;
  for(int i=0;i<length;i++)
  {
    for(int j=0;j<width;j++)
    {
      data=pica.at<uchar>(i,j);
      data=data<<1;
      data=data+picb.at<uchar>(i,j);
      data=data<<1;
      data=data+picc.at<uchar>(i,j);
      data=data<<1;
      data=data+picd.at<uchar>(i,j);
      data=data<<1;
      data=data+pice.at<uchar>(i,j);
      data=data<<1;
      data=data+picf.at<uchar>(i,j);
      data=data<<1;
      data=data+picg.at<uchar>(i,j);
      data=data<<1;
      data=data+pich.at<uchar>(i,j);
      Image.at<uchar>(i,j)=data;
    }
  }
}
void negative(Mat inputMat)
{
  int length = inputMat.cols;
  int width = inputMat.rows;
  for(int i=0;i<length;i++)
  {
    for(int j=0;j<width;j++)
    {
      if(inputMat.at<uchar>(i,j)==0)
        inputMat.at<uchar>(i,j)=1;
      else if(inputMat.at<uchar>(i,j)==1)
        inputMat.at<uchar>(i,j)=0;
    }
  }
}
void transbit(Mat inputMat)
{
  int length = inputMat.cols;
  int width = inputMat.rows;
  for(int i=0;i<length;i++)
  {
    for(int j=0;j<width;j++)
    {
      if(inputMat.at<uchar>(i,j)==255)
        inputMat.at<uchar>(i,j)=1;
      else if(inputMat.at<uchar>(i,j)==0)
        inputMat.at<uchar>(i,j)=0;
    }
  }
}

void logtrans(Mat inputMat,Mat outputMat,int c)
{
  int length = inputMat.cols;
  int width = inputMat.rows;
  float data=0;
  for(int i=0;i<length;i++)
  {
    for(int j=0;j<width;j++)
    {
      data=(float)inputMat.at<uchar>(i,j)/255;
      outputMat.at<uchar>(i,j) = c * log(1+data);
    }
  }
}
void powerLaw(Mat inputMat,Mat outputMat,float g,int c)
{
  int length = inputMat.cols;
  int width = inputMat.rows;
  float data;
  for(int i=0;i<length;i++)
  {
    for(int j=0;j<width;j++)
    {
      data = (float)inputMat.at<uchar>(i,j)/255;
      outputMat.at<uchar>(i,j) = c * pow(data,g);
    }
  }
}
void histogram(Mat inputMat,Mat outputMat)
{
  int length = inputMat.cols;
  int width = inputMat.rows;
  long his[256];
  long max = 0;
  float data=0;
  for(int i=0;i<256;i++)
    his[i]=0;
  for(int i=0;i<length;i++)
  {
    for(int j=0;j<width;j++)
      his[inputMat.at<uchar>(i,j)]++;
  }
  for(int i=0;i<256;i++)
  {
    if(his[i]>max)
      max = his[i];
  }
  for(int i=0;i<256;i++)
  {
    data = ((float)his[i]*255)/max;  //0~1  >>0~255

    for(int j=0;j<256;j++)
    {
      if((255-j)<data)
      {
        outputMat.at<uchar>(j,i)=0;
      }
      else
      {
        outputMat.at<uchar>(j,i)=255;
      }
    }
  }
}
void equal(Mat inputMat,Mat outputMat)
{
  int length = inputMat.cols;
  int width = inputMat.rows;
  long his[256];
  long count=0;
  int trans[256];
  long data=0;
  for(int i=0;i<256;i++)
    his[i]=0;
  for(int i=0;i<length;i++)
  {
    for(int j=0;j<width;j++)
    {
      his[inputMat.at<uchar>(i,j)]++;
      count++;
    }
  }
  for(int i=0;i<256;i++)
  {
    data=data+his[i];
    trans[i]=data*255/count;
  }
  for(int i=0;i<length;i++)
  {
    for(int j=0;j<width;j++)
    {
      outputMat.at<uchar>(i,j)=trans[inputMat.at<uchar>(i,j)];
    }
  }
  

  
}
//主程式開始
int main() {      
  //-------變數宣告區------------------------------------------------
  unsigned char rawfile[256*256];
  unsigned char rawfilebig[512*512];
  string inputstring;
  //-------Mat宣告區------------------------------------------------
  Mat baboon_256(256,256,CV_8UC1);
  Mat baboon_tran(256,256,CV_8UC1);
  Mat lena_256(256,256,CV_8UC1);
  Mat lena_tran(256,256,CV_8UC1);
  Mat test(256,256,CV_8UC1);
  Mat test2(256,256,CV_8UC1);
  Mat merge(256,256,CV_8UC1);
  Mat origin(512,512,CV_8UC1);
  Mat livingroom_d(512,512,CV_8UC1);
  Mat cameraman_b(512,512,CV_8UC1);
  Mat livingroom(512,512,CV_8UC1);
  Mat livingroom1(512,512,CV_8UC1);
  Mat cameraman(512,512,CV_8UC1);
  Mat cameraman1(512,512,CV_8UC1);
  Mat his_living(256,256,CV_8UC1);
  Mat his_camera(256,256,CV_8UC1);
  Mat living_tran(512,512,CV_8UC1);
  Mat camera_tran(512,512,CV_8UC1);
  Mat a(512,512,CV_8UC1);
  Mat b(512,512,CV_8UC1);
  Mat c(512,512,CV_8UC1);
  Mat d(512,512,CV_8UC1);
  Mat e(512,512,CV_8UC1);
  Mat f(512,512,CV_8UC1);
  Mat g(512,512,CV_8UC1);
  Mat h(512,512,CV_8UC1);
  //-------Mat設定區------------------------------------------------

  //-------讀取檔案區------------------------------------------------
  Read_Raw("../data/input/baboon_256.raw",rawfile,256,256);
  for(int i=0;i<baboon_256.rows;i++)
  {
    for(int j=0;j<baboon_256.cols;j++)
      baboon_256.at<uchar>(i,j)=rawfile[i*256+j];
  }
  Read_Raw("../data/input/lena_256.raw",rawfile,256,256);
  for(int i=0;i<lena_256.rows;i++)
  {
    for(int j=0;j<lena_256.cols;j++)
      lena_256.at<uchar>(i,j)=rawfile[i*256+j];
  }
  Read_Raw("../data/input/a512x512.raw",rawfilebig,512,512);
  for(int i=0;i<a.rows;i++)
  {
    for(int j=0;j<a.cols;j++)
    {
      a.at<uchar>(i,j)=rawfilebig[i*512+j];
      //printf("%d,",a.at<uchar>(i,j));
    }
  }
  Read_Raw("../data/input/b512x512.raw",rawfilebig,512,512);
  for(int i=0;i<b.rows;i++)
  {
    for(int j=0;j<b.cols;j++)
      b.at<uchar>(i,j)=rawfilebig[i*512+j];
  }
  Read_Raw("../data/input/c512x512.raw",rawfilebig,512,512);
  for(int i=0;i<c.rows;i++)
  {
    for(int j=0;j<c.cols;j++)
      c.at<uchar>(i,j)=rawfilebig[i*512+j];
  }
  Read_Raw("../data/input/d512x512.raw",rawfilebig,512,512);
  for(int i=0;i<d.rows;i++)
  {
    for(int j=0;j<d.cols;j++)
      d.at<uchar>(i,j)=rawfilebig[i*512+j];
  }
  Read_Raw("../data/input/e512x512.raw",rawfilebig,512,512);
  for(int i=0;i<e.rows;i++)
  {
    for(int j=0;j<e.cols;j++)
      e.at<uchar>(i,j)=rawfilebig[i*512+j];
  }
  Read_Raw("../data/input/f512x512.raw",rawfilebig,512,512);
  for(int i=0;i<f.rows;i++)
  {
    for(int j=0;j<f.cols;j++)
      f.at<uchar>(i,j)=rawfilebig[i*512+j];
  }
  Read_Raw("../data/input/g512x512.raw",rawfilebig,512,512);
  for(int i=0;i<g.rows;i++)
  {
    for(int j=0;j<g.cols;j++)
      g.at<uchar>(i,j)=rawfilebig[i*512+j];
  }
  Read_Raw("../data/input/h512x512.raw",rawfilebig,512,512);
  for(int i=0;i<h.rows;i++)
  {
    for(int j=0;j<h.cols;j++)
      h.at<uchar>(i,j)=rawfilebig[i*512+j];
  }
  Read_Raw("../data/input/livingroom_d512.raw",rawfilebig,512,512);
  for(int i=0;i<b.rows;i++)
  {
    for(int j=0;j<b.cols;j++)
      livingroom_d.at<uchar>(i,j)=rawfilebig[i*512+j];
  }
  Read_Raw("../data/input/cameraman_b512.raw",rawfilebig,512,512);
  for(int i=0;i<b.rows;i++)
  {
    for(int j=0;j<b.cols;j++)
      cameraman_b.at<uchar>(i,j)=rawfilebig[i*512+j];
  }
  
  imwrite("../data/output/lena_256.png",lena_256);
  imwrite("../data/output/baboon_256.png",baboon_256);

  //--------程式開始----------------------------------------------
  printf("please enter the question number \n enter quit to exit\n");
  printf("menu \n 3-1-1 3-1-2\n 3-2-a\n 3-3-a 3-3-b-liv 3-3-b-cam\n");
  printf(" 3-4-a 3-4-b\n");

  while(inputstring != "quit") 
  {
    printf("please enter the question number \n");
    cin >> inputstring;
    if(inputstring=="3-1-1")
    {
      MergeImage(lena_256,baboon_256,merge,5);
      printf("lena_with_baboon  MSE=%f,PSNR=%f\n",MSE(lena_256,merge),PSNR(lena_256,merge));
      for(int i=0;i<merge.cols;i++)
      {
        for(int j=0;j<merge.rows;j++)
        {
          rawfile[i*256+j]=merge.at<uchar>(i,j);
        }
      }
      Write_Raw("../data/output/3-1-1-lena_with_baboon.raw",rawfile,256,256);
      imwrite("../data/output/3-1-1-lena_with_baboon.png",merge);
      imshow("lena_with_baboon",merge);
      waitKey(0);
      destroyAllWindows();
    }
    else if(inputstring=="3-1-2")
    {
      SplitImage(merge,lena_tran,baboon_tran,5);
      printf("lena  MSE=%f,PSNR=%f\n",MSE(lena_tran,lena_256),PSNR(lena_tran,lena_256));
      printf("baboon  MSE=%f,PSNR=%f\n",MSE(baboon_tran,baboon_256),PSNR(baboon_tran,baboon_256));
      imshow("lena_extract",lena_tran);
      imshow("baboon_extract",baboon_tran);
      imwrite("../data/output/3-1-2-lena_extract.png",lena_tran);
      imwrite("../data/output/3-1-2-baboon_extract.png",baboon_tran);
      waitKey(0);
      destroyAllWindows();
    }
    else if(inputstring=="3-2-a")
    {
      transbit(a);
      transbit(b);
      transbit(c);
      transbit(d);
      transbit(e);
      transbit(f);
      transbit(g);
      transbit(h);

      negative(h);
      negative(e);
      negative(d);

      mixture(origin,b,g,h,c,a,e,d,f);
      for(int i=0;i<origin.cols;i++)
      {
        for(int j=0;j<origin.rows;j++)
        {
          rawfile[i*512+j]=origin.at<uchar>(i,j);
        }
      }
      Write_Raw("../data/output/origin.raw",rawfilebig,512,512);
      imwrite("../data/output/origin.png",origin);
      imshow("origin",origin);
      waitKey(0);
      destroyAllWindows();
    }
    else if(inputstring=="3-3-a")
    {
      printf("first c=100 than c=20\n");

      logtrans(livingroom_d,livingroom,100);
      logtrans(cameraman_b,cameraman,100);
      imshow("livingroom c=100",livingroom);
      imshow("cameraman c=100",cameraman);
      imwrite("../data/output/3-3a-livingroom_log_c100.png",livingroom);
      imwrite("../data/output/3-3a-cameraman_log_c100.png",cameraman);
      waitKey(0);
      destroyAllWindows();

      logtrans(livingroom_d,livingroom,20);
      logtrans(cameraman_b,cameraman,20);
      imshow("livingroom c=20",livingroom);
      imshow("cameraman c=20",cameraman);
      imwrite("../data/output/3-3a-livingroom_log_c20.png",livingroom);
      imwrite("../data/output/3-3a-cameraman_log_c20.png",cameraman);
      waitKey(0);
      destroyAllWindows();
    }
    else if(inputstring=="3-3-b-liv")
    {
      powerLaw(livingroom_d,livingroom,0.2,255);
      powerLaw(livingroom_d,livingroom1,0.2,100);
      imshow("livingroom gamma = 0.2,c=255",livingroom);
      imshow("livingroom gamma = 0.2,c=100",livingroom1);
      imwrite("../data/output/3-3b-livingroom_gamma02_c255.png",livingroom);
      imwrite("../data/output/3-3b-livingroom_gamma02_c100.png",livingroom1);
      waitKey(0);
      destroyAllWindows();

      powerLaw(livingroom_d,livingroom,10,255);
      powerLaw(livingroom_d,livingroom1,10,100);
      imshow("livingroom gamma = 10,c=255",livingroom);
      imshow("livingroom gamma = 10,c=100",livingroom1);
      imwrite("../data/output/3-3b-livingroom_gamma10_c255.png",livingroom);
      imwrite("../data/output/3-3b-livingroom_gamma10_c100.png",livingroom1);
      waitKey(0);
      destroyAllWindows();

    }
    else if(inputstring=="3-3-b-cam")
    {
      powerLaw(cameraman_b,cameraman,0.2,255);
      powerLaw(cameraman_b,cameraman1,0.2,100);
      imshow("cameraman gamma = 0.2,c=100",cameraman1);
      imshow("cameraman gamma = 0.2,c=255",cameraman);
      imwrite("../data/output/3-3b-cameraman_gamma02_c255.png",cameraman);
      imwrite("../data/output/3-3b-cameraman_gamma02_c100.png",cameraman1);
      waitKey(0);  
      destroyAllWindows();    

      powerLaw(cameraman_b,cameraman,10,255);
      powerLaw(cameraman_b,cameraman1,10,100);
      imshow("cameraman gamma = 10,c=255",cameraman);
      imshow("cameraman gamma = 10,c=100",cameraman1);
      imwrite("../data/output/3-3b-cameraman_gamma10_c255.png",cameraman);
      imwrite("../data/output/3-3b-cameraman_gamma10_c100.png",cameraman1);
      waitKey(0);
      destroyAllWindows(); 
    }
    else if(inputstring=="3-4-a")
    {
      histogram(livingroom_d,his_living);
      histogram(cameraman_b,his_camera);
      imshow("histogram_living",his_living);
      imshow("histogram_camera",his_camera);
      imwrite("../data/output/3-4a-livigroom_histogram.png",his_living);
      imwrite("../data/output/3-4a-cameraman_histogram.png",his_camera);
      waitKey(0);
      destroyAllWindows();
    }
    else if(inputstring=="3-4-b")
    {
      equal(cameraman_b,camera_tran);
      equal(livingroom_d,living_tran);

      imshow("camera_tran",camera_tran);
      imshow("living_tran",living_tran);
      waitKey(0);
      destroyAllWindows();  
      imwrite("../data/output/3-4b-livigroom_tran.png",living_tran);
      imwrite("../data/output/3-4b-cameraman_tran.png",camera_tran);

      histogram(living_tran,his_living);
      histogram(camera_tran,his_camera);
      imshow("histogram_living",his_living);
      imshow("histogram_camera",his_camera);
      imwrite("../data/output/3-4b-livigroom_histogram_eq.png",his_living);
      imwrite("../data/output/3-4b-cameraman_histogram_eq.png",his_camera);
      waitKey(0);
      destroyAllWindows();  
    }

  }




  return 0;
}
