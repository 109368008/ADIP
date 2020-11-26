#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <time.h>

using namespace cv;


int main() {

  FILE *fp;
	if(NULL == (fp = fopen("../data/lena_256.raw", "r")))
	{
		printf("error\n");
	    exit(1);
	}
 
	int ch;                                
  unsigned char pic[256][256];            //圖檔大小 256*256*8
  unsigned int i=0,j=0,k=0;//,data=0;
  unsigned int x=0,y=0,key=0;
	while(EOF != (ch=fgetc(fp)))            //每次讀取一個字元（8bit） 直到最後一個字元
	{
    
    pic[j][i]=ch;                         //將資料排入矩陣

    /*x=(j*256)+i;                        //紀錄第5487個資料
    if(x==5487)
    {
      data=pic[i][j];
    }*/
    if(i != 255)
      i++;
    else
    {
      i=0;
      j++;
    }
	}

	fclose(fp);

  
//----輸出讀取數值------------------------------------
  /*for(i=0;i<255;i++)                    
  {
    for(j=0;j<255;j++)
    {
      printf("%d ",pic[i][j]);
    }
  }*/
//--------輸出答案---------------------------------------------
  printf("HW1-1.2.b-(1) : \n");
  printf("row=123,col=234 value = %d",pic[234][123]);
  printf("\n");
  printf("HW1-1.2.b-(2) : \n");
  printf("5487th pixel coordinate is ( %d , %d )\n",5487/256,5487%256);  //取5487除256之餘數為X,商為Y
//-------顯示圖片-----------------------------

  Mat Image,Image2;
  Image2.create( 256,256, CV_8UC1);
  Image.create( 256,256, CV_8UC1);
  namedWindow("HW1-1.2-c",WINDOW_AUTOSIZE );
  for(i=0;i<Image.rows;i++)
  {
    for(j=0;j<Image.cols;j++)
    {
      Image.at<uchar>(j, i) = pic[j][i];
    }
  }
  printf("show lmage 'HW1-1.2-c' \n");
  imshow("HW1-1.2-c",Image);
  while(key != 'q')
    key = waitKey();
  key=0;
  destroyWindow("HW1-1.2-c");

  namedWindow("HW1-1.2-d",WINDOW_AUTOSIZE );
  for(i=0;i<Image.rows;i++)
  {
    for(j=0;j<Image.cols;j++)
    {
      Image.at<uchar>(i, j) = pic[j][i]; //將 row 與 col 對調 可以使照片旋轉90度
    }
  }
  printf("show lmage 'HW1-1.2-d' \n");
  imshow("HW1-1.2-d",Image);
  while(key != 'q')
    key = waitKey();
  key=0;
  destroyWindow("HW1-1.2-d");   
//產生0~15之亂數陣列
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
  namedWindow("HW1-1.2-e1",WINDOW_AUTOSIZE );
  namedWindow("HW1-1.2-e2",WINDOW_AUTOSIZE );
  for(i=0;i<Image.rows;i++)           
  {
    for(j=0;j<Image.cols;j++)
    {
      y=(rnd1[(i/64)*4+j/64]%4)*64;
      x=(rnd1[(i/64)*4+j/64]/4)*64;
      Image.at<uchar>(j, i) = pic[j%64+y][i%64+x];
      y=(rnd2[(i/64)*4+j/64]%4)*64;
      x=(rnd2[(i/64)*4+j/64]/4)*64;
      Image2.at<uchar>(j, i) = pic[j%64+y][i%64+x];    

    }
  }
  printf("show lmage 'HW1-1.2-e1' \n");
  imshow("HW1-1.2-e1",Image);
  printf("show lmage 'HW1-1.2-e2' \n");
  imshow("HW1-1.2-e2",Image2);
  while(key != 'q')
    key = waitKey();
  key=0;
  destroyWindow("HW1-1.2-e1");
  destroyWindow("HW1-1.2-e2");
//------1.2-f------------------------------
  unsigned char pall[4][256][256];        //圖檔大小 256*256*8 共有4個
  i=0;
  j=0;
	if(NULL == (fp = fopen("../data/p01.raw", "r")))
	{
		printf("error\n");
	    exit(1);
	}                           
  
	while(EOF != (ch=fgetc(fp)))            //每次讀取一個字元（8bit） 直到最後一個字元
	{
    pall[3][j][i]=ch;                         //將資料排入矩陣
    if(i != 255)
      i++;
    else
    {
      i=0;
      j++;
    }
	}
	fclose(fp);
  i=0;
  j=0;
	if(NULL == (fp = fopen("../data/p02.raw", "r")))
	{
		printf("error\n");
	    exit(1);
	}                           
	while(EOF != (ch=fgetc(fp)))            //每次讀取一個字元（8bit） 直到最後一個字元
	{
    pall[2][j][i]=ch;                         //將資料排入矩陣
    if(i != 255)
      i++;
    else
    {
      i=0;
      j++;
    }
	}
	fclose(fp);
  i=0;
  j=0;
  	if(NULL == (fp = fopen("../data/p03.raw", "r")))
	{
		printf("error\n");
	    exit(1);
	}                           
	while(EOF != (ch=fgetc(fp)))            //每次讀取一個字元（8bit） 直到最後一個字元
	{
    pall[0][j][i]=ch;                         //將資料排入矩陣
    if(i != 255)
      i++;
    else
    {
      i=0;
      j++;
    }
	}
	fclose(fp);
  i=0;
  j=0;
  	if(NULL == (fp = fopen("../data/p04.raw", "r")))
	{
		printf("error\n");
	    exit(1);
	}                           
	while(EOF != (ch=fgetc(fp)))            //每次讀取一個字元（8bit） 直到最後一個字元
	{
    pall[1][j][i]=ch;                         //將資料排入矩陣
    if(i != 255)
      i++;
    else
    {
      i=0;
      j++;
    }
	}
	fclose(fp);

  Image.create( 512,512, CV_8UC1);
  namedWindow("HW1-1.2-f",WINDOW_AUTOSIZE );
  unsigned char moon[512][512];
  for(k=0;k<4;k++)
  {
    for(i=0;i<Image.rows/2;i++)
    {
      for(j=0;j<Image.cols/2;j++)
      {
        x=(k%2)*256;
        y=(k/2)*256;
        Image.at<uchar>(j+y, i+x) = pall[k][j][i]; 
        moon[j+y][i+x]= pall[k][j][i];
      }
    }
  }
  printf("show lmage 'HW1-1.2-f' \n");
  imshow("HW1-1.2-f",Image);
  while(key != 'q')
    key = waitKey();
  key=0;
    
  destroyWindow("HW1-1.2-f");  

  if(NULL == (fp = fopen("../data/moon.raw", "w")))
	{
		printf("error\n");
	    exit(1);
	} 
  for(i=0;i<512;i++)
  {
    for(j=0;j<512;j++)
    {
      fprintf(fp,"%c",moon[i][j]);
    }
  }
  fclose (fp);

//----1.3_a-----------------

  unsigned char lena_50[256][256];
  unsigned char lena_r50[256][256];
  int data=0;
  for(i=0;i<256;i++)
  {
    for(j=0;j<256;j++)
    {
      data = pic[j][i]+50; //將 row 與 col 對調 可以使照片旋轉90度
      if(data>255)
        data=255;
      lena_50[j][i]=data;
      data = pic[j][i]+(rand()%100)-50; 
      if(data<0)
        data=0;
      else if(data>255)
        data=255;
      lena_r50[j][i]=data;
    }
  }

  if(NULL == (fp = fopen("../data/lena+50.raw", "w")))
	{
		printf("error\n");
	    exit(1);
	} 
  for(i=0;i<256;i++)
  {
    for(j=0;j<256;j++)
    {
      fprintf(fp,"%c",lena_50[i][j]);
    }
  }
  fclose (fp);
  printf("output HW1-1.3-a");
  if(NULL == (fp = fopen("../data/lena+r50.raw", "w")))
	{
		printf("error\n");
	    exit(1);
	} 
  for(i=0;i<256;i++)
  {
    for(j=0;j<256;j++)
    {
      fprintf(fp,"%c",lena_r50[i][j]);
    }
  }
  fclose (fp);
  printf("output HW1-1.3-b");


/*
  for(i=0;i<16;i++)
  {
    printf("%d ",rnd1[i]);
  } 
  printf("\n");

  for(i=0;i<16;i++)
  {
    printf("%d ",rnd2[i]);
  } 
  printf("\n");
*/
  i=0;
  j=0;
  unsigned int haha[512][512];
  if(NULL == (fp = fopen("../data/haha.raw", "r")))
	{
		printf("error\n");
	    exit(1);
	}                           
  
	while(EOF != (ch=fgetc(fp)))            //每次讀取一個字元（8bit） 直到最後一個字元
	{
    haha[j][i]=ch;                         //將資料排入矩陣
    if(i != 511)
      i++;
    else
    { 
      i=0;
      j++;
    }
	}
	fclose(fp);

  namedWindow("HW1-2",WINDOW_AUTOSIZE);
  for(i=0;i<Image.rows;i++)
  {
    for(j=0;j<Image.cols;j++)
    {
      Image.at<uchar>(j, i) = haha[j][i]; 
    }
  }
  putText(Image,"109368008",Point2i(100,45),FONT_HERSHEY_SIMPLEX,2,0,2,10,false);
  putText(Image,"NICE",Point2i(180,470),FONT_HERSHEY_SIMPLEX,3,0,10,10,false);

  printf("show lmage 'HW1-2' \n");
  imshow("HW1-2",Image);
  while(key != 'q')
    key = waitKey();
  key=0;
  imwrite("../data/HW1-2.png",Image);
  destroyWindow("HW1-2");  
  









  return 0;
}