#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

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
//主程式開始
int main() {                          
  unsigned int i=0,j=0,k=0;             //回圈會使用之變數
  unsigned int x=0,y=0,key=0,part=0;    //座標平移,鍵值之變數
  unsigned char pic_lena[256*256];      //圖檔大小 256*256*8bit      
  unsigned char pic_all[512*512];       //圖檔大小 256*256*8bit  共有4個 =512*512
  unsigned char moon[512*512];          //儲存拼接後圖檔空間
  unsigned char lena_50[256*256];       //add 50 value
  unsigned char lena_r50[256*256];      //add -50~50 random value
  unsigned char haha[512*512];          //haha檔案
  int data=0;                           //1.3-ab
  unsigned char randomcut1[256*256];
  unsigned char randomcut2[256*256];
  string inputstring;

//-------顯示圖片1.2-c--------------------------------------------------------
  cv::Mat Image1_2c,Image1_2d,Image1_2e1,Image1_2e2,Image1_2f,Image1_3a,Image1_3b,Image2;  
  Image1_2c.create( 256,256, CV_8UC1);
  Image1_2d.create( 256,256, CV_8UC1);
  Image1_2e1.create( 256,256, CV_8UC1);
  Image1_2e2.create( 256,256, CV_8UC1);
  Image1_2f.create( 512,512, CV_8UC1);
  Image1_3a.create( 256,256, CV_8UC1);
  Image1_3b.create( 256,256, CV_8UC1);
  Image2.create( 512,512, CV_8UC1);

  Read_Raw("../data/input/lena_256.raw",pic_lena,256,256);
  for(i=0;i<Image1_2c.rows;i++)
  {
    for(j=0;j<Image1_2c.cols;j++)
      Image1_2c.at<uchar>(j, i) = pic_lena[j*256+i];
  }
//-------顯示圖片1.2-d------------------------------------------------------------------------
  for(i=0;i<Image1_2d.rows;i++)
  {
    for(j=0;j<Image1_2d.cols;j++)
      Image1_2d.at<uchar>(i, j) = pic_lena[j*256+i]; //將 row 與 col 對調 可以使照片旋轉90度
  }
//產生0~15之亂數陣列---------------------------------------------
  char rnd1[16],rnd2[16];
  srand(time(NULL) );
  for(i=0;i<16;i++)
  {
      rnd1[i] = rand()%16;
      for(j=0;j<i;j++)
      {
        while(rnd1[i]==rnd1[j])
        {
          j=0;
          rnd1[i]=rand()%16;
        }
      }
      rnd2[i] = rand()%16;
      for(j=0;j<i;j++)
      {
        while(rnd2[i]==rnd2[j])
        {
          j=0;
          rnd2[i]=rand()%16;
        }
      }
  }
//-----顯示圖片1.2-e---------------------------------------------------
  for(i=0;i<Image1_2e1.rows;i++)           
  {
    for(j=0;j<Image1_2e1.cols;j++)
    {
      y=(rnd1[(i/64)*4+j/64]%4)*64;
      x=(rnd1[(i/64)*4+j/64]/4)*64;
      Image1_2e1.at<uchar>(j, i) = pic_lena[((j%64+y)*256)+i%64+x];
      randomcut1[j*256+i]=pic_lena[((j%64+y)*256)+i%64+x];
      y=(rnd2[(i/64)*4+j/64]%4)*64;
      x=(rnd2[(i/64)*4+j/64]/4)*64;
      Image1_2e2.at<uchar>(j, i) = pic_lena[((j%64+y)*256)+i%64+x];    
      randomcut2[j*256+i]=pic_lena[((j%64+y)*256)+i%64+x];
    }
  }
  Write_Raw("../data/output/HW1-1.2.e_rnd1.raw",randomcut1,256,256);
  Write_Raw("../data/output/HW1-1.2.e_rnd2.raw",randomcut2,256,256);
//------1.2-f------------------------------
  Read_Raw("../data/input/p03.raw",pic_all,256,256);              //讀入第1張影像
  Read_Raw("../data/input/p04.raw",pic_all+(256*256),256,256);    //接續讀入第2張影像
  Read_Raw("../data/input/p02.raw",pic_all+(256*256)*2,256,256);  //接續讀入第3張影像
  Read_Raw("../data/input/p01.raw",pic_all+(256*256)*3,256,256);  //接續讀入第4張影像
  for(k=0;k<4;k++)
  {
      part=k*256*256;
      x=k%2*256;
      y=k/2*256;
    for(i=0;i<Image1_2f.rows/2;i++)
    {
      for(j=0;j<Image1_2f.cols/2;j++)
      {
        Image1_2f.at<uchar>(j+y,i+x) = pic_all[part+(j*256)+i]; //(k*256*256)+
        moon[(j+y)*512+(i+x)]= pic_all[part+(j*256)+i];
      }
    }
  }
  Write_Raw("../data/output/HW1-1.2-f.raw",moon,512,512);
//----1.3_ab-----------------------------------------------------------------
  for(i=0;i<65536;i++)
  {
    data = pic_lena[i]+50;
    if(data>255)
      data=255;
    lena_50[i]=data;
    Image1_3a.at<uchar>(i/256,i%256)=data;
    data = pic_lena[i]+(rand()%100)-50;
    if(data<0)
      data=0;
    else if(data>255)
      data=255;
    lena_r50[i]=data;
    Image1_3b.at<uchar>(i/256,i%256)=data;
  }
  Write_Raw("../data/output/HW1-1.3-1.raw",lena_50,256,256);
  Write_Raw("../data/output/HW1-1.3-2.raw",lena_r50,256,256);
//------HW1-2----------------------------------------------------------------
  Read_Raw("../data/input/haha.raw",haha,512,512);
  
  for(i=0;i<Image2.rows;i++)
  {
    for(j=0;j<Image2.cols;j++)
      Image2.at<uchar>(j, i) = haha[j*512+i]; 
  }
  putText(Image2,"109368008",Point2i(100,45),FONT_HERSHEY_SIMPLEX,2,0,2,10,false);
  putText(Image2,"NICE",Point2i(180,470),FONT_HERSHEY_SIMPLEX,3,0,10,10,false);
  //------------------程式文字介面------------------------------------
  printf("please enter the question number \n enter quit to exit\n");
  printf("menu \n 1-2b 1-2c 1-2d 1-2e 1-2f\n 1-3a 1-3b\n 2-1\n");
  while(inputstring != "quit") 
  {
    printf("please enter the question number \n");
    cin >> inputstring;
    if(inputstring=="1-2b")
    {
      printf("HW1-1.2.b-(1) : \n");
      printf("row=123,col=234 value = %d\n",pic_lena[(123*256)+234]); //二維座標轉一維座標
      printf("HW1-1.2.b-(2) : \n");
      printf("5487th pixel coordinate is ( %d , %d ) and value is %d\n",5487/256,5487%256,pic_lena[(5487/256)*256+5487%256]);  //取5487除256之餘數為X,商為Y
    }
    else if(inputstring=="1-2c")
    {
      namedWindow("HW1-1.2-c",WINDOW_AUTOSIZE );
      imshow("HW1-1.2-c",Image1_2c);
      waitKey(0);
      destroyWindow("HW1-1.2-c");
    }
    else if(inputstring=="1-2d")
    {
      namedWindow("HW1-1.2-d",WINDOW_AUTOSIZE ); 
      imshow("HW1-1.2-d",Image1_2d);
      waitKey(0);
      destroyWindow("HW1-1.2-d"); 
    }
    else if(inputstring=="1-2e")
    {
      namedWindow("HW1-1.2-e1",WINDOW_AUTOSIZE ); 
      namedWindow("HW1-1.2-e2",WINDOW_AUTOSIZE ); 
      imshow("HW1-1.2-e1",Image1_2e1);
      imshow("HW1-1.2-e2",Image1_2e2);
      waitKey(0);
      destroyWindow("HW1-1.2-e1"); 
      destroyWindow("HW1-1.2-e2"); 
    }
    else if(inputstring=="1-2f")
    {
      namedWindow("HW1-1.2-f",WINDOW_AUTOSIZE ); 
      imshow("HW1-1.2-f",Image1_2f);
      waitKey(0);
      destroyWindow("HW1-1.2-f"); 
    }
    else if(inputstring=="1-3a")
    {
      namedWindow("HW1-1.3-a",WINDOW_AUTOSIZE);
      imshow("HW1-1.3-a",Image1_3a);
      waitKey(0);
      destroyWindow("HW1-1.3-a");   
    }
    else if(inputstring=="1-3b")
    {
      namedWindow("HW1-1.3-b",WINDOW_AUTOSIZE);
      imshow("HW1-1.3-b",Image1_3b);
      waitKey(0);
      destroyWindow("HW1-1.3-b");   
    }
    else if(inputstring=="2-1")
    {
      namedWindow("HW1-2-1",WINDOW_AUTOSIZE);
      imshow("HW1-2-1",Image2);
      waitKey(0);
      destroyWindow("HW1-2-1"); 
    }
  }
  return 0;
}