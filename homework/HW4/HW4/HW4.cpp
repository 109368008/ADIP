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
void histogram(Mat inputMat,unsigned long* hist)
{
  int length = inputMat.cols;
  int width = inputMat.rows;
  int h;
  for(int i=0;i<256;i++)
    *(hist+i)=0;
  for(int i=0;i<length;i++)
  {
    for(int j=0;j<width;j++)
    {
      h=inputMat.at<uchar>(i,j);
      *(hist+h)= *(hist+h)+1;
    }
  }
}
void cdf(unsigned long* hist,unsigned int* cdl,unsigned long size)
{
  float data=0;
  for(int i=0;i<256;i++)
     data = ((float)*(hist+i)*255)/size;
  data=0;
  for(int i=0;i<256;i++)
  {
    data=data+*(hist+i);
    *(cdl+i)=data*255/size;
  }
}
void drawhistogram(Mat inputMat,Mat outputMat)
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
        outputMat.at<uchar>(j,i)=0;
      else
        outputMat.at<uchar>(j,i)=255;
    }
  }
}
void equal(Mat inputMat,Mat outputMat,unsigned int* cdl)
{
  int length = inputMat.cols;
  int width = inputMat.rows;
  long size =length*width;
  for(int i=0;i<length;i++)
  {
    for(int j=0;j<width;j++)
      outputMat.at<uchar>(i,j)=*(cdl+inputMat.at<uchar>(i,j));
  }
}
void cdf_inv(unsigned int* cdl)
{
  unsigned int trans[256];
  for(int i=0;i<256;i++)
  {
    trans[i]= 0;
  }
  for(int i=0;i<256;i++)
  {
    trans[*(cdl+i)]= i;
  }
  for(int i=0;i<256;i++)
  {
    if(trans[i]==0)
      trans[i]=trans[i-1];
  }  
  for(int i=0;i<256;i++)
  {
    *(cdl+i)=trans[i];
  }
}
void padding(Mat inputMat,Mat outputMat,int mask,char mode) //mode 0=zero mode 1 = same
{
  int length = inputMat.cols;
  int width = inputMat.rows; 
  int half=(mask-1)/2;
  
  for(int i=0;i<outputMat.cols;i++)
  {
    for(int j=0;j<outputMat.rows;j++)
    {
      if(i<half or j<half)
      {
        if(mode==0)
          outputMat.at<uchar>(i,j)=0;
        else if(mode==1)
        {
          if(i<half and j<half)
            outputMat.at<uchar>(i,j)=inputMat.at<uchar>(0,0);
          else if(i<half)
            outputMat.at<uchar>(i,j)=inputMat.at<uchar>(0,j-half);
          else if(j<half)
            outputMat.at<uchar>(i,j)=inputMat.at<uchar>(i-half,0);
        }
        else
        {
          if(i<half and j<half)
            outputMat.at<uchar>(i,j)=inputMat.at<uchar>(half-i,half-j);
          else if(i<half)
            outputMat.at<uchar>(i,j)=inputMat.at<uchar>(half-i,j-half);
          else if(j<half)
            outputMat.at<uchar>(i,j)=inputMat.at<uchar>(i-half,half-j);
        }
      }
      else if (i>length-1+half or j>width-1+half)
      {
        if(mode==0)
          outputMat.at<uchar>(i,j)=0;
        else if(mode==1)
        {
          if(i>length-1+half and j>width-1+half)
            outputMat.at<uchar>(i,j)=inputMat.at<uchar>(length-1,width-1);
          else if(i>length-1+half)
            outputMat.at<uchar>(i,j)=inputMat.at<uchar>(length-1,j);
          else if(j>width-1+half)
            outputMat.at<uchar>(i,j)=inputMat.at<uchar>(i,width-1);
        }
        else
        {
          if(i>length-1+half and j>width-1+half)
            outputMat.at<uchar>(i,j)=inputMat.at<uchar>(2*(length-1)-i+half,2*(width-1)-j+half);
          else if(i>length-1+half)
            outputMat.at<uchar>(i,j)=inputMat.at<uchar>(2*(length-1)-i+half,j-half);
          else if(j>width-1+half)
            outputMat.at<uchar>(i,j)=inputMat.at<uchar>(i-half,2*(width-1)-j+half);
        }
      }
      else
        outputMat.at<uchar>(i,j)=inputMat.at<uchar>(i-half,j-half);    
    }
  } 
}
void localeq(Mat inputMat,Mat outputMat,int mask,char mode)
{
  
  int length = inputMat.cols;
  int width = inputMat.rows;
  long his[256];
  long count=0;
  int trans[256];
  long data=0;
  int half=(mask-1)/2;
  count=mask*mask;
  Mat tranMat(length+2*half,width+2*half,CV_8UC1);
  padding(inputMat,tranMat,mask,mode);
  for(int i=half;i<length+half;i++)
  {
    for(int j=half;j<width+half;j++)
    {
      
      for(int k=0;k<256;k++)
        his[k]=0;
      for(int ii=i-half;ii<=i+half;ii++)
      {
        for(int jj=j-half;jj<=j+half;jj++)
          his[tranMat.at<uchar>(ii,jj)]++;
      }    
      data=0;
      for(int k=0;k<256;k++)
      {
        data = data+his[k];
        trans[k] = data*255/count;
      }
      outputMat.at<uchar>(i-half,j-half)=trans[tranMat.at<uchar>(i,j)];
    }
  }
}
void kernelfilter(Mat inputMat,Mat outputMat,float kernal[],unsigned char kernelsize,char mode)
{ 
  int length = inputMat.rows;
  int width = inputMat.cols;
  float ker[kernelsize][kernelsize]={0};
  int half=(kernelsize-1)/2;
  float data=0;
  Mat trans(length+2*half,width+2*half,CV_8UC1);
  padding(inputMat,trans,kernelsize,mode);
  for(int i=0;i<kernelsize;i++)
  {
    for(int j=0;j<kernelsize;j++)
    {
        ker[i][j]=kernal[(i*kernelsize)+j];
    }
  }

  for(int i=half;i<length+half;i++)
  {
    for(int j=half;j<width+half;j++)
    {
      data=0;
      for(int ii=i-half;ii<=i+half;ii++)
      {
        for(int jj=j-half;jj<=j+half;jj++)
          data=data+(ker[ii-i+half][jj-j+half]*trans.at<uchar>(ii,jj));    
      }
      if(data<0)
        data=0;
      else if(data>255)
        data=255;
      outputMat.at<uchar>(i-half,j-half)=data;
    }
  }
}
void gaussian(Mat inputMat,Mat outputMat,float sigmoid1,float sigmoid2,int size)
{
  int pi = M_PI;
  float gkernel[size*size]={0};
  float gkernel1[size][size]={0};
  float gkernel2[size][size]={0};
  int half=(size-1)/2;
  float data=0;
  float sigmoid=0;
  for(int i=0-half;i<=half;i++)
  {
    for(int j=0-half;j<=half;j++)
    {
      sigmoid=sigmoid1;
      gkernel1[i+half][j+half]=(1/(2*pi*pow(sigmoid,2)))*exp((0-(pow(i,2)+pow(j,2)))/(2*pow(sigmoid,2)));
      if(sigmoid2 !=0)
      {
        sigmoid=sigmoid2;
        gkernel2[i+half][j+half]=(1/(2*pi*pow(sigmoid,2)))*exp((0-(pow(i,2)+pow(j,2)))/(2*pow(sigmoid,2)));
      }
    }
  }
  for(int i=0;i<size;i++)
  {
    for(int j=0;j<size;j++)
    {
      gkernel[(i)*size+j]=gkernel1[i][j]-gkernel2[i][j];
    }
  }
  kernelfilter(inputMat,outputMat,gkernel,size,2);
  int max=0;
  int min=255;
  int num=0;
  int range = 0;
  for(int i=0;i<outputMat.rows;i++)
  {
    for(int j=0; j<outputMat.cols;j++)
    {
      num=outputMat.at<uchar>(i,j);
      if(num>max)
        max=num;
      if(num<min)
        min=num;
    }
  }
  range=max-min;
  for(int i=0;i<outputMat.rows;i++)
  {
    for(int j=0; j<outputMat.cols;j++)
    {
      num=(outputMat.at<uchar>(i,j)-min)*255/range;
      outputMat.at<uchar>(i,j)=num;
    }
  }
}

//主程式開始
int main() {      
  //-------變數宣告區------------------------------------------------
  unsigned char rawfile[256*256];
  unsigned char rawfilebig[512*512];
  unsigned long list[256];
  unsigned int cdf1[256];
  string inputstring;
  //-------Mat宣告區------------------------------------------------
  Mat house512(512,512,CV_8UC1);
  Mat house512_eq(512,512,CV_8UC1);
  Mat lena512(512,512,CV_8UC1);
  Mat lena_his(256,256,CV_8UC1);
  Mat house_his(256,256,CV_8UC1);
  Mat test(512,512,CV_8UC1);
  Mat test2(512,512,CV_8UC1);
  Mat walkbridge(512,512,CV_8UC1);
  Mat turtle512(512,512,CV_8UC1);
  Mat turtle512_08(512,512,CV_8UC1);
  Mat turtle512_13(512,512,CV_8UC1);
  Mat turtle512_20(512,512,CV_8UC1);
  Mat hist(256,256,CV_8UC1);
  //-------讀取檔案區------------------------------------------------
  Read_Raw("../data/input/house512.raw",rawfilebig,512,512);
  for(int i=0;i<house512.rows;i++)
  {
    for(int j=0;j<house512.cols;j++)
      house512.at<uchar>(i,j)=rawfilebig[i*512+j];
  }
  Read_Raw("../data/input/lena512.raw",rawfilebig,512,512);
  for(int i=0;i<lena512.rows;i++)
  {
    for(int j=0;j<lena512.cols;j++)
      lena512.at<uchar>(i,j)=rawfilebig[i*512+j];
  }
  Read_Raw("../data/input/walkbridge.raw",rawfilebig,512,512);
  for(int i=0;i<walkbridge.rows;i++)
  {
    for(int j=0;j<walkbridge.cols;j++)
      walkbridge.at<uchar>(i,j)=rawfilebig[i*512+j];
  }
  Read_Raw("../data/input/turtle512.raw",rawfilebig,512,512);
  for(int i=0;i<turtle512.rows;i++)
  {
    for(int j=0;j<turtle512.cols;j++)
      turtle512.at<uchar>(i,j)=rawfilebig[i*512+j];
  }
  
  imwrite("../data/output/turtle512.png",turtle512);
  imwrite("../data/output/house512.png",house512);
  imwrite("../data/output/walkbridge.png",walkbridge);
  imwrite("../data/output/lena512.png",lena512);

  printf("please enter the question number \n enter quit to exit\n");
  printf("menu \n 4-1a 4-1b \n 4-2_10p 4-2_200p 4-2_all \n");
  printf(" 4-3a 4-3b\n 4-4a 4-4b\n");

  while(inputstring != "quit") 
  {
    printf("please enter the question number \n");
    cin >> inputstring;
    if(inputstring=="4-1b")
    {
      Mat his_4_1_b(256,256,CV_8UC1);
      printf("4-1-b-kernel=3\n");
      localeq(house512,test,3,1);
      drawhistogram(test,his_4_1_b);
      imwrite("../data/output/4-1-b-kernel=3_his.png",his_4_1_b);
      imwrite("../data/output/4-1-b-kernel=3.png",test);
      imshow("4-1-b-kernel=3_his",his_4_1_b);
      imshow("4-1-b-kernel=3",test);
      waitKey(0);
      destroyAllWindows();

      printf("4-1-b-kernel=5\n");
      localeq(house512,test,5,1);
      drawhistogram(test,his_4_1_b);
      imwrite("../data/output/4-1-b-kernel=5_his.png",his_4_1_b);
      imwrite("../data/output/4-1-b-kernel=5.png",test);
      imshow("4-1-b-kernel=5_his",his_4_1_b);
      imshow("4-1-b-kernel=5",test);
      waitKey(0);
      destroyAllWindows();

      printf("4-1-b-kernel=7\n");
      localeq(house512,test,7,1);
      drawhistogram(test,his_4_1_b);
      imwrite("../data/output/4-1-b-kernel=7_his.png",his_4_1_b);
      imwrite("../data/output/4-1-b-kernel=7.png",test);
      imshow("4-1-b-kernel=7_his",his_4_1_b);
      imshow("4-1-b-kernel=7",test);
      waitKey(0);
      destroyAllWindows();
    }
    else if(inputstring=="4-1a")
    {
      unsigned int lena_cdf[256];
      unsigned int house_cdf[256];
      unsigned long size = 512*512;
      histogram(lena512,list);
      cdf(list,lena_cdf,size);
      histogram(house512,list);
      cdf(list,house_cdf,size);
      equal(house512,house512_eq,house_cdf);
      cdf_inv(lena_cdf);
      equal(house512_eq,test2,lena_cdf);
      
      drawhistogram(lena512,lena_his);
      drawhistogram(test2,house_his);
      drawhistogram(house512,hist);
      imwrite("../data/output/4-1-a-house_histogram.png",hist);
      imwrite("../data/output/4-1-a-house_match_to_lena.png",test2);
      imwrite("../data/output/4-1-a-lena_histogram.png",lena_his);
      imwrite("../data/output/4-1-a-house_histogram_match_to_lena.png",house_his);
      imshow("house_histogram",hist);
      imshow("house_match_to_lena",test2);
      imshow("lena_histogram",lena_his);
      imshow("house_histogram_match_to_lena",house_his);
      waitKey(0);
      destroyAllWindows();
    }
    else if(inputstring=="4-2_10p")
    {
      int pics=10;
      VideoCapture cap("../data/input/street.avi");
      Mat avrimage(540,960,CV_8UC1);
      if (!cap.isOpened()) {
          cout << "Error opening video" << endl;
          return -1;
      }

      long count=0;
      int length;
      int width;
      double avrframe[960][540]={0};
      
      while (1) {
          Mat frame;
          //下方程式碼會依序獲得每一帧
          //請根據上課所學修改此段
          cap >> frame;
          length=960;
          width=540;
          //若影片結束跳出迴圈
          if (frame.empty())
              break;
          cvtColor(frame,frame,COLOR_BGR2GRAY);
          //printf("%d,%d\n",frame.cols,frame.rows);
          //imshow("Frame", frame);
          for(int i=0;i<length;i++)
          {
            for(int j=0;j<width;j++)
              avrframe[i][j]=avrframe[i][j]+frame.at<uchar>(j,i);         
          }
          count++;
          if(count==pics)
            break;
          char c = (char)waitKey(25);
          if (c == 27)
              break;
      }
      for(int i=0;i<length;i++)
      {
        for(int j=0;j<width;j++)        
          avrimage.at<uchar>(j,i)= avrframe[i][j]/count;
      }
      imwrite("../data/output/4-2_10p.png",avrimage);
      imshow("avr",avrimage);
      waitKey(0);
      cap.release();
      destroyAllWindows();
    }
    else if(inputstring=="4-2_200p")
    {
      int pics=200;
      VideoCapture cap("../data/input/street.avi");
      Mat avrimage(540,960,CV_8UC1);
      if (!cap.isOpened()) {
          cout << "Error opening video" << endl;
          return -1;
      }

      long count=0;
      int length;
      int width;
      double avrframe[960][540]={0};
      
      while (1) {
          Mat frame;
          //下方程式碼會依序獲得每一帧
          //請根據上課所學修改此段
          cap >> frame;
          length=960;
          width=540;
          //若影片結束跳出迴圈
          if (frame.empty())
              break;
          cvtColor(frame,frame,COLOR_BGR2GRAY);
          //printf("%d,%d\n",frame.cols,frame.rows);
          //imshow("Frame", frame);
          for(int i=0;i<length;i++)
          {
            for(int j=0;j<width;j++)
              avrframe[i][j]=avrframe[i][j]+frame.at<uchar>(j,i);         
          }
          count++;
          if(count==pics)
            break;
          //printf("%lu\n",count);
          char c = (char)waitKey(25);
          if (c == 27)
              break;
      }
      //printf("%d,%d",length,width);

      for(int i=0;i<length;i++)
      {
        for(int j=0;j<width;j++)
        {          
          avrimage.at<uchar>(j,i)= avrframe[i][j]/count;
        }
      }

      imshow("avr",avrimage);
      imwrite("../data/output/4-2_200p.png",avrimage);
      waitKey(0);
      cap.release();
      destroyAllWindows();
    }
    else if(inputstring=="4-2_all")
    {
      VideoCapture cap("../data/input/street.avi");
      Mat avrimage(540,960,CV_8UC1);
      if (!cap.isOpened()) {
          cout << "Error opening video" << endl;
          return -1;
      }
      long count=0;
      int length;
      int width;
      double avrframe[960][540]={0};
      while (1) {
          Mat frame;
          //下方程式碼會依序獲得每一帧
          //請根據上課所學修改此段
          cap >> frame;
          length=960;
          width=540;
          //若影片結束跳出迴圈
          if (frame.empty())
              break;
          cvtColor(frame,frame,COLOR_BGR2GRAY);
          //printf("%d,%d\n",frame.cols,frame.rows);
          //imshow("Frame", frame);
          for(int i=0;i<length;i++)
          {
            for(int j=0;j<width;j++)
              avrframe[i][j]=avrframe[i][j]+frame.at<uchar>(j,i);         
          }
          count++;
          //printf("%lu\n",count);
          char c = (char)waitKey(25);
          if (c == 27)
              break;
      }
      for(int i=0;i<length;i++)
      {
        for(int j=0;j<width;j++)        
          avrimage.at<uchar>(j,i)= avrframe[i][j]/count;
      }
      imshow("avr",avrimage);
      imwrite("../data/output/4-2_all.png",avrimage);
      waitKey(0);
      cap.release();
      destroyAllWindows();
    }
    else if(inputstring=="4-3a")
    {
      Mat walkbridge_edge1(512,512,CV_8UC1);
      Mat walkbridge_edge2(512,512,CV_8UC1);
      Mat walkbridge_edge3(512,512,CV_8UC1);
      Mat walkbridge_edge4(512,512,CV_8UC1);
      float filter1[]={0,-1,0,-1,4,-1,0,-1,0};
      float filter2[]={-1,-1,-1,-1,8,-1,-1,-1,-1};
      kernelfilter(walkbridge,walkbridge_edge1,filter1,3,0);
      imwrite("../data/output/4-3a-walkbridge_edge_zero_nocorner.png",walkbridge_edge1);
      kernelfilter(walkbridge,walkbridge_edge2,filter1,3,1);
      imwrite("../data/output/4-3a-walkbridge_edge_same_nocorner.png",walkbridge_edge2);
      kernelfilter(walkbridge,walkbridge_edge3,filter2,3,0);
      imwrite("../data/output/4-3a-walkbridge_edge_zero_corner.png",walkbridge_edge3);
      kernelfilter(walkbridge,walkbridge_edge4,filter2,3,1);
      imwrite("../data/output/4-3a-walkbridge_edge_same_corner.png",walkbridge_edge4);
      imshow("4-3a-walkbridge_edge_zero_nocorner",walkbridge_edge1);
      imshow("4-3a-walkbridge_edge_same_nocorner",walkbridge_edge2);
      imshow("4-3a-walkbridge_edge_zero_corner",walkbridge_edge3);
      imshow("4-3a-walkbridge_edge_same_corner",walkbridge_edge4);
      waitKey(0);
      destroyAllWindows();
    }
    else if(inputstring=="4-3b")
    {
      float filter1[]={0,-1,0,-1,8,-1,0,-1,0};
      float filter2[]={-1,-1,-1,-1,12,-1,-1,-1,-1};
      Mat walkbridge_edge1(512,512,CV_8UC1);
      Mat walkbridge_edge2(512,512,CV_8UC1);
      Mat walkbridge_edge3(512,512,CV_8UC1);
      Mat walkbridge_edge4(512,512,CV_8UC1);
      string name1,name2;
      for(int i=0;i<3;i++)
      {
        if(i==0)
        {
          filter1[4]=5;
          filter2[4]=9;
          name1="4-3b-walkbridge_edge_replicated_nocorner_A=1";
          name2="4-3b-walkbridge_edge_replicated_corner_A=1";
        }
        else if(i==1)
        {
          filter1[4]=6;
          filter2[4]=10;
          name1="4-3b-walkbridge_edge_replicated_nocorner_A=2";
          name2="4-3b-walkbridge_edge_replicated_corner_A=2";
        }
        else
        {
          filter1[4]=8;
          filter2[4]=12;
          name1="4-3b-walkbridge_edge_replicated_nocorner_A=4";
          name2="4-3b-walkbridge_edge_replicated_corner_A=4";
        }
        kernelfilter(walkbridge,walkbridge_edge1,filter1,3,1);
        kernelfilter(walkbridge,walkbridge_edge2,filter2,3,1);
        imshow(name1,walkbridge_edge1);
        imshow(name2,walkbridge_edge2);
        imwrite("../data/output/"+name1+".png",walkbridge_edge1);
        imwrite("../data/output/"+name2+".png",walkbridge_edge2);
        waitKey(0);
        destroyAllWindows();
      }
    }
    else if(inputstring=="4-4a")
    {
      gaussian(turtle512,turtle512_08,0.8,0,5);
      gaussian(turtle512,turtle512_13,1.3,0,5);
      gaussian(turtle512,turtle512_20,2.0,0,5);
      imwrite("../data/output/4-4a-turtle=08.png",turtle512_08);
      imwrite("../data/output/4-4a-turtle=13.png",turtle512_13);
      imwrite("../data/output/4-4a-turtle=20.png",turtle512_20);
      imshow("4-4a-turtle=0.8",turtle512_08);
      imshow("4-4a-turtle=1.3",turtle512_13);
      imshow("4-4a-turtle=2.0",turtle512_20);
      waitKey(0);
      destroyAllWindows();
    }
    else if(inputstring=="4-4b")
    {
      Mat turtle_dog(512,512,CV_8UC1);
      gaussian(turtle512,turtle_dog,0.5,1.5,5);
      imwrite("../data/output/4-4b-turtle-dog.png",turtle_dog);
      imshow("origin",turtle512);
      imshow("4-4b-turtle-dog",turtle_dog);
      waitKey(0);
      destroyAllWindows(); 
    }
  }
  return 0;
}
