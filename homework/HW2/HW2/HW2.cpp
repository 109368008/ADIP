#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <time.h>
#include <math.h>
#define MAXSTACK 400 /*定義最大堆疊容量*/

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
void Zoomin_part(unsigned int x1,unsigned int y1,unsigned int x2,unsigned int y2,cv::Mat inputMat,cv::Mat outputMat,unsigned int magnification,int shift)//unsigned char* outputMat
{
  int error=0;
  if(x1>=x2)
    error=1;
  else if(y1>=y2)
    error=2;
  else if(x2>inputMat.cols)
    error=3;
  else if(y2>inputMat.rows)
    error=4;
  if(error != 0)
  {
    printf("error %d",error);
    return;
  }

  unsigned int length=x2-x1;
  unsigned int width=y2-y1;
  unsigned int startx=((x2+x1)/2)-(length*magnification/2)+shift;
  unsigned int starty=((y2+y1)/2)-(width*magnification/2);
  unsigned char part[width][length];

  for(int i=0;i<length;i++)
  {
    for(int j=0;j<width;j++)
      part[j][i]= inputMat.at<uchar>(y1+j,x1+i);
  }
  for(int i=0;i<length*magnification;i++)
  {
    for(int j=0;j<width*magnification;j++)
      outputMat.at<uchar>(starty+j,startx+i)=part[j/magnification][i/magnification];
  }
}
void bilinear(Mat inputMat,Mat outputMat)
{ 
  float shrinkx,shrinky;
  shrinkx=(float)outputMat.rows/inputMat.rows;
  shrinky=(float)outputMat.cols/inputMat.cols;
  //printf("%f,%f\n",shrinkx,shrinky);
  int x1,x2,y1,y2;
  float x,y;
  int sum1,sum2,sum3;
  for(int i=0;i<outputMat.cols;i++)
  {
    for(int j=0;j<outputMat.rows;j++)
    {
      x=j/shrinkx;
      y=i/shrinky;
      x1=floor(x);
      x2=floor(x)+1;
      y1=floor(y);
      y2=floor(y)+1;
      sum1 = (x2-x)/(x2-x1)*inputMat.at<uchar>(x1,y1)+(x-x1)/(x2-x1)*inputMat.at<uchar>(x2,y1);
      sum2 = (x2-x)/(x2-x1)*inputMat.at<uchar>(x1,y2)+(x-x1)/(x2-x1)*inputMat.at<uchar>(x2,y2);
      sum3 = (y2-y)/(y2-y1)*sum1+(y-y1)/(y2-y1)*sum2+0.5;
      outputMat.at<uchar>(j,i)=sum3;//inputMat.at<uchar>(j/shrinkx,i/shrinky);
      //printf("%d,",sum3);
    }
  }
}
float Interpolate(int a0,int a1,int a2,int a3,float x)
{
  float p1,p2,p3,p4,ps;
  p1=(float)(a3-(2*a2)+(2*a1)-a0)/6;
  p2=(float)(a0-a1)/2;
  p3=(float)(a3-(8*a2)+(5*a1)+(2*a0))/(-6);
  p4=(float)(a1);
  ps=(p1*x*x*x)+(p2*x*x)+(p3*x)+p4;
  return ps;
}
void bicubic(Mat inputMat,Mat outputMat)
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
void neighbor(Mat inputMat,Mat outputMat)
{ 
  float shrinkx,shrinky;
  shrinkx=(float)outputMat.rows/inputMat.rows;
  shrinky=(float)outputMat.cols/inputMat.cols;
  for(int i=0;i<outputMat.cols;i++)
  {
    for(int j=0;j<outputMat.rows;j++)
    {
      int x,y;
      x=j/shrinkx+0.5;
      y=i/shrinky+0.5;
      outputMat.at<uchar>(j,i)=inputMat.at<uchar>(x,y);
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
void delay(int number_of_seconds) 
{ 
    // Converting time into milli_seconds 
    int milli_seconds =  number_of_seconds; 
  
    // Storing start time 
    clock_t start_time = clock(); 
  
    // looping till required time is not achieved 
    while (clock() < start_time + milli_seconds) 
        ; 
} 

Mat fullpath;
Mat tempath;
int dis=-1,temp=-1;
int stack[MAXSTACK];  //堆疊的陣列宣告 
int top=-1;		//堆疊的頂端
/*判斷是否為空堆疊*/
int isEmpty(){
	if(top==-1){
		return 1; 
	}else{
		return 0;
	}
} 
/*將指定的資料存入堆疊*/
void push(int data)
{
	if(top>=MAXSTACK){
		printf("堆疊已滿,無法再加入\n");	
	}else{
		top++;
		stack[top]=data;
	}
} 
/*從堆疊取出資料*/
int pop()
{
	int data;
	if(isEmpty())
  {
		printf("堆疊已空\n");
    return -1;
	}
  else{
		data=stack[top];
    top--;
		return data; 
	}
}
int Read_stack_top()
{
  int data;
  if(isEmpty())
  {
		printf("堆疊已空\n");
    return -1;
	}
  else
  {
		data=stack[top];
		return data; 
	}
}
int Search_stack(int x)
{
  int data;
  int i;
  if(isEmpty())
  {
		printf("堆疊已空\n");
    return -1;
  }
  else
  {
    for(i=top;i>0;i--)
    {
      if(stack[i]==x)
        return i;
    }
    return -1;
  }
}

unsigned int fin=0;

int P4(int x,int y,Mat map,Mat path,int vet)//map and path is 0 1 value  ,unsigned char v[]
{
  int length=map.cols,width=map.rows;
  int count,cross=0,dead=0;
  char move[4]={0,0,0,0};
  char m[width][length];
  char p[width][length];
  int startx,starty,startv;
  startx=x;
  starty=y;
  startv=vet;
  for(int i=0;i<width;i++)
  {
    for(int j=0;j<length;j++)
    {
      m[i][j]=map.at<uchar>(i,j);
      p[i][j]=path.at<uchar>(i,j);
      m[i][j]=m[i][j]-p[i][j];
    }
  }  

  while(1)
  {
    p[y][x]=1;
    cross=0;
    count=0;
    for(int i=-1;i<2;i++)
    {
      for(int j=-1;j<2;j++)
      {
        
        if(i*i==j*j)          //no corner
          continue;
        if(y+i<0 || x+j<0 || y+i>19 || x+j>19)    //no out data
        {
          move[count]=0;
          count++;
          continue;
        }
        if(p[y+i][x+j]== 1) // is path
          move[count]=0;
        else if(m[y+i][x+j]== 1 )
        {
          move[count]=1;
          cross++;
        }
        else 
          move[count]=0;
        count++;
      }
    }
    if(cross==0)
    {
      if(x==19 && y==19)
      {
        if(fin==0)
        {
          dis=-1;
          for(int i=0;i<width;i++)
          {
            for(int j=0;j<length;j++)
            {
              fullpath.at<uchar>(i,j)=(path.at<uchar>(i,j)+p[i][j])*255;
              if(fullpath.at<uchar>(i,j)!=0)
                dis++;
            }
          } 
          fin++;
          dead=1;
        }
        else
        {
          temp=-1;
          for(int i=0;i<width;i++)
          {
            for(int j=0;j<length;j++)
            {
              if(path.at<uchar>(i,j)+p[i][j]!=0)
                temp++;
            }
          } 
          if(temp<dis)
          {
            for(int i=0;i<width;i++)
            {
              for(int j=0;j<length;j++)
              {
                fullpath.at<uchar>(i,j)=(path.at<uchar>(i,j)+p[i][j])*255;
                if(fullpath.at<uchar>(i,j)!=0)
                  temp++;
              }
            } 
          }
          fin++;
          dead=1;
        }
      }
      else 
      {
        dead=1;
      }
    }
    else if(cross==1)
    {
      for(int i=0;i<=4;i++)
      {
        if(move[i]==1)
        {
          switch (i)
          {
            case 0:
              y=y-1;
              break;
            case 1:
              x=x-1;
              break;
            case 2:
              x=x+1;
              break;
            case 3:
              y=y+1;
              break;
          }
        }
      }
    }
    else 
    {
      for(int i=0;i<4;i++)
      {
        if(move[i]==1)
        {
          switch (i)
          {
            case 0:
              y=y-1;
              vet=0;
              break;
            case 1:
              x=x-1;
              vet=1;
              break;
            case 2:
              x=x+1;
              vet=2;
              break;
            case 3:
              y=y+1;
              vet=3;
              break;
          }
          for(int i=0;i<width;i++)
          {
            for(int j=0;j<length;j++)
            {
              path.at<uchar>(i,j)=p[i][j];
            }
          } 
          switch(P4(x,y,map,path,vet))
          {
            case 0:
              y=y+1;
              break;
            case 1:
              x=x+1;
              break;
            case 2:
              x=x-1;
              break;
            case 3:
              y=y-1;
              break;
            default:
              break;
          }
        }
        if(i==3)
          dead=1;
      }
    }
    if(dead != 0)
      break;
  }
  return startv;
}

int P8(int x,int y,Mat map,Mat path,int vet)//map and path is 0 1 value  ,unsigned char v[]
{
  int length=map.cols,width=map.rows;
  int count,cross=0,dead=0;
  char move[8]={0,0,0,0,0,0,0,0};
  char m[width][length];
  char p[width][length];
  int startx,starty,startv;
  startx=x;
  starty=y;
  startv=vet;
  for(int i=0;i<width;i++)
  {
    for(int j=0;j<length;j++)
    {
      m[i][j]=map.at<uchar>(i,j);
      p[i][j]=path.at<uchar>(i,j);
      m[i][j]=m[i][j]-p[i][j];
    }
  }  

  while(1)
  {
    p[y][x]=1;
    cross=0;
    count=0;
    for(int i=-1;i<2;i++)
    {
      for(int j=-1;j<2;j++)
      {
        if(i==0 && j==0)
          continue;
        if(y+i<0 || x+j<0 || y+i>19 || x+j>19)    //no out data
        {
          move[count]=0;
          count++;
          continue;
        }
        if(p[y+i][x+j]== 1) // is path
          move[count]=0;
        else if(m[y+i][x+j]== 1 )
        {
          move[count]=1;
          cross++;
        }
        else 
          move[count]=0;
        count++;
      }
    }
    if(cross==0)
    {
      if(x==19 && y==19)
      {
        if(fin==0)
        {
          dis=-1;
          for(int i=0;i<width;i++)
          {
            for(int j=0;j<length;j++)
            {
              fullpath.at<uchar>(i,j)=(path.at<uchar>(i,j)+p[i][j])*255;
              if(fullpath.at<uchar>(i,j)!=0)
                dis++;
            }
          } 
          fin++;
          dead=1;
        }
        else
        {
          temp=-1;
          for(int i=0;i<width;i++)
          {
            for(int j=0;j<length;j++)
            {
              if(path.at<uchar>(i,j)+p[i][j]!=0)
                temp++;
            }
          } 
          if(temp<dis)  
          {
            for(int i=0;i<width;i++)
            {
              for(int j=0;j<length;j++)
              {
                fullpath.at<uchar>(i,j)=(path.at<uchar>(i,j)+p[i][j])*255;
                dis=temp;
              }
            } 
            //printf("dis=%d\n",dis);
          }
          fin++;
          dead=1;
        }

      }
      else 
      {
        dead=1;
      }
    }
    else if(cross==1)
    {
      for(int i=0;i<=8;i++)
      {
        if(move[i]==1)
        {
          switch (i)
          {
            case 0:
              y=y-1;
              x=x-1;
              break;
            case 1:
              y=y-1;
              break;
            case 2:
              y=y-1;
              x=x+1;
              break;
            case 3:
              x=x-1;
              break;
            case 4:
              x=x+1;
              break;
            case 5:
              x=x-1;
              y=y+1;
              break;
            case 6:
              y=y+1;
              break;
            case 7:
              x=x+1;
              y=y+1;
              break;
          }
        }
      }
    }
    else 
    {
      for(int i=0;i<8;i++)
      {
        if(move[i]==1)
        {
          switch (i)
          {
            case 0:
              y=y-1;
              x=x-1;
              vet=0;
              break;
            case 1:
              y=y-1;
              vet=1;
              break;
            case 2:
              y=y-1;
              x=x+1;
              vet=2;
              break;
            case 3:
              x=x-1;
              vet=3;
              break;
            case 4:
              x=x+1;
              vet=4;
              break;
            case 5:
              x=x-1;
              y=y+1;
              vet=5;
              break;
            case 6:
              y=y+1;
              vet=6;
              break;
            case 7:
              x=x+1;
              y=y+1;
              vet=7;
              break;
          }
          for(int i=0;i<width;i++)
          {
            for(int j=0;j<length;j++)
            {
              path.at<uchar>(i,j)=p[i][j];
            }
          } 
          switch(P8(x,y,map,path,vet))
          {
            case 0:
              y=y+1;
              x=x+1;
              break;
            case 1:
              y=y+1;
              break;
            case 2:
              y=y+1;
              x=x-1;
              break;
            case 3:
              x=x+1;
              break;
            case 4:
              x=x-1;
              break;
            case 5:
              x=x+1;
              y=y-1;
              break;
            case 6:
              y=y-1;
              break;
            case 7:
              x=x-1;
              y=y-1;
              break;
            default:
              break;
          }
        }
        if(i==3)
          dead=1;
      }
    }
    if(dead != 0)
      break;
  }
 
  return startv;
}



//主程式開始
int main() {       
  unsigned char rawfile[256*256];
  clock_t start_t,end_t;
  float total_t;
  int key=0;
  string inputstring;
  Mat lena_256,lena_bigeyes,lena_128,lena_blur;
  Mat lena_bl1,lena_bl2,lena_bl3;
  Mat lena_nn1,lena_nn2,lena_nn3;
  Mat lena_bc1,lena_bc2,lena_bc3,pic_trans;
  Mat baboon_256;
  Mat map,fmap;
  Mat path;
  Mat pic32,pic64,pic128,pic256;
  Mat tran32,tran64,tran128,tran256;
  tran32.create(32,32,CV_8UC1);
  tran64.create(64,64,CV_8UC1);
  tran128.create(128,128,CV_8UC1);
  tran256.create(256,256,CV_8UC1); 
  pic32.create(32,32,CV_8UC1);
  pic64.create(64,64,CV_8UC1);
  pic128.create(128,128,CV_8UC1);
  pic256.create(256,256,CV_8UC1);

  path.create(20,20,CV_8UC1);
  lena_256.create(256,256,CV_8UC1);
  lena_blur.create(256,256,CV_8UC1);
  baboon_256.create(256,256,CV_8UC1);
  map.create(20,20,CV_8UC1);
  fmap.create(20,20,CV_8UC1);
  fullpath.create(20,20,CV_8UC1);
  tempath.create(20,20,CV_8UC1);

  Read_Raw("../data/input/map.raw",rawfile,20,20);
  for(int i=0;i<map.rows;i++)
  {
    for(int j=0;j<map.cols;j++)
      fmap.at<uchar>(i,j)=rawfile[i*20+j];
  }
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
  Read_Raw("../data/input/lena_256_result25.raw",rawfile,256,256);
  for(int i=0;i<lena_blur.rows;i++)
  {
    for(int j=0;j<lena_blur.cols;j++)
      lena_blur.at<uchar>(i,j)=rawfile[i*256+j];
  }
  lena_bigeyes=lena_256.clone();



  printf("please enter the question number \n enter quit to exit\n");
  printf("menu \n 2-1-1 \n 2-1-2\n 2-1-3a 2-1-3b 2-1-3c\n 2-2\n 2-3lena 2-3baboon\n ");
  printf("2-5-1-d4 2-5-2-d4 2-5-3-d4\n 2-5-1-d8 2-5-2-d8 2-5-3-d8\n");
  printf("2-5-3-d8 's execution time is over 12 hour\n");
  printf("Dm doesn't finished\n");
  while(inputstring != "quit") 
  {
    printf("please enter the question number \n");
    cin >> inputstring;
    if(inputstring == "2-1-1")
    {
      //--------   1-1  --------------------------------------------
      Zoomin_part(123,128,143,139,lena_256,lena_bigeyes,2,-3); //左眼
      Zoomin_part(158,128,175,139,lena_256,lena_bigeyes,2,3); //右眼

      namedWindow("HW2-1-1");
      imshow("HW2-1-1",lena_bigeyes);
      waitKey(0);
      destroyWindow("HW2-1-1");
      imwrite("../data/output/lena_bigeyes.png",lena_bigeyes);
    }
    else if(inputstring == "2-1-2")
    {
    //-----------  1-2 -----------------------------------------------
      lena_128.create(128,128,CV_8UC1);
      neighbor(lena_blur,lena_128);
      namedWindow("HW2-1-2");
      imshow("HW2-1-2",lena_128);
      waitKey(0);
      destroyWindow("HW2-1-2");
      imwrite("../data/output/lena_128.png",lena_128);
    }
    else if(inputstring == "2-1-3a")
    {
    //-------near neighbor-----
    //----------- ^2.5 v2  ----------------------  
      start_t=clock();
      pic_trans.create(640,640,CV_8UC1);
      neighbor(lena_blur,pic_trans);
      lena_nn1.create(320,320,CV_8UC1);
      neighbor(pic_trans,lena_nn1);
      end_t=clock();
      total_t=(float)end_t-start_t;
      printf("nearneighbor ^2.5v2 = %f us\n",total_t);
      imwrite("../data/output/nearneighbor1.png",lena_nn1);
    //------------  v2 ^2.5 --------------------------
      start_t=clock();
      pic_trans.create(128,128,CV_8UC1);
      neighbor(lena_blur,pic_trans);
      lena_nn2.create(320,320,CV_8UC1);
      neighbor(pic_trans,lena_nn2);
      end_t=clock();
      total_t=(float)end_t-start_t;
      printf("nearneighbor v2^2.5 = %f us\n",total_t);
      imwrite("../data/output/nearneighbor2.png",lena_nn2);
    //-------------  ^1.25   --------------------------
      start_t=clock();
      lena_nn3.create(320,320,CV_8UC1);
      neighbor(lena_blur,lena_nn3);
      end_t=clock();
      total_t=(float)end_t-start_t;
      printf("nearneighbor ^1.25 = %f us\n",total_t);
      imwrite("../data/output/nearneighbor3.png",lena_nn3);
      namedWindow("nearneighbor ^2.5v2");
      namedWindow("nearneighbor v2^2.5");
      namedWindow("nearneighbor ^1.25");
      imshow("nearneighbor ^2.5v2",lena_nn1);
      imshow("nearneighbor v2^2.5",lena_nn2);
      imshow("nearneighbor ^1.25",lena_nn3);
      waitKey(0);
      destroyWindow("nearneighbor ^2.5v2");
      destroyWindow("nearneighbor v2^2.5");
      destroyWindow("nearneighbor ^1.25");
    }
    else if(inputstring == "2-1-3b")
    {
    //-------bilinear-----
    //----------- ^2.5 v2  ----------------------  
      start_t=clock();
      pic_trans.create(640,640,CV_8UC1);
      bilinear(lena_blur,pic_trans);
      lena_bl1.create(320,320,CV_8UC1);
      bilinear(pic_trans,lena_bl1);
      end_t=clock();
      total_t=(float)end_t-start_t;
      printf("bilinear ^2.5v2 = %f us\n",total_t);
      imwrite("../data/output/bilinear1.png",lena_bl1);
    //-----------v2 ^2.5   --------------------
      start_t=clock();
      pic_trans.create(128,128,CV_8UC1);
      bilinear(lena_blur,pic_trans);
      lena_bl2.create(320,320,CV_8UC1);
      bilinear(pic_trans,lena_bl2);
      end_t=clock();
      total_t=(float)end_t-start_t;
      printf("bilinear v2^2.5 = %f us\n",total_t);
      imwrite("../data/output/bilinear2.png",lena_bl2);
    //------------- ^1.25 ---------------------
      start_t=clock();
      lena_bl3.create(320,320,CV_8UC1);
      bilinear(lena_blur,lena_bl3);
      end_t=clock();
      total_t=(float)end_t-start_t;
      printf("bilinear ^1.25 = %f us\n",total_t);
      imwrite("../data/output/bilinear3.png",lena_bl3);
      
      namedWindow("bilinear ^2.5v2");
      namedWindow("bilinear v2^2.5");
      namedWindow("bilinear ^1.25");
      imshow("bilinear ^2.5v2",lena_bl1);
      imshow("bilinear v2^2.5",lena_bl2);
      imshow("bilinear ^1.25",lena_bl3);
      waitKey(0);
      destroyWindow("bilinear ^2.5v2");
      destroyWindow("bilinear v2^2.5");
      destroyWindow("bilinear ^1.25");
    }
    else if(inputstring == "2-1-3c")
    {
    //-------------bicubic-----------------------
    //----------- ^2.5 v2  ----------------------  
      start_t=clock();
      pic_trans.create(640,640,CV_8UC1);
      bicubic(lena_blur,pic_trans);
      lena_bc1.create(320,320,CV_8UC1);
      bicubic(pic_trans,lena_bc1);
      end_t=clock();
      total_t=(float)end_t-start_t;
      printf("bicubic ^2.5v2 = %f us\n",total_t);
      imwrite("../data/output/bc1.png",lena_bc1);
      
    //-----------v2 ^2.5   --------------------
      start_t=clock();
      pic_trans.create(128,128,CV_8UC1);
      bicubic(lena_blur,pic_trans);
      lena_bc2.create(320,320,CV_8UC1);
      bicubic(pic_trans,lena_bc2);
      end_t=clock();
      total_t=(float)end_t-start_t;
      printf("bicubic v2^2.5 = %f us\n",total_t);
      imwrite("../data/output/bc2.png",lena_bc2);
    //------------- ^1.25 ---------------------
      start_t=clock();
      lena_bc3.create(320,320,CV_8UC1);
      bicubic(lena_blur,lena_bc3);
      end_t=clock();
      total_t=(float)end_t-start_t;
      printf("bicubic ^1.25 = %f us\n",total_t);
      imwrite("../data/output/bc3.png",lena_bc3);
      
      namedWindow("bicubic ^2.5v2");
      namedWindow("bicubic v2^2.5");
      namedWindow("bicubic ^1.25");
      imshow("bicubic ^2.5v2",lena_bc1);
      imshow("bicubic v2^2.5",lena_bc2);
      imshow("bicubic ^1.25",lena_bc3);
      waitKey(0);
      destroyWindow("bicubic ^2.5v2");
      destroyWindow("bicubic v2^2.5");
      destroyWindow("bicubic ^1.25");
    }
    else if(inputstring == "2-2")
    {
      printf("show lena first then baboon \n 8bits -> 1bits resolution\n");
      pic_trans.create(256,256,CV_8UC1);
      for(int i=8;i>0;i--)
      {
        string filenum="../data/output/GLR_lena_"+to_string(i)+".png";
        Gray_level_resolution(lena_256,pic_trans,i);
        imwrite(filenum,pic_trans);
        namedWindow("Gray_level_resolution");
        imshow("Gray_level_resolution",pic_trans);
        printf("MSE of lena Gray_level_resolution of %d bit is %f \n",i,MSE(lena_256,pic_trans));
        printf("PSNR of lena Gray_level_resolution of %d bit is %f \n",i,PSNR(lena_256,pic_trans));
        waitKey(0);
        destroyWindow("Gray_level_resolution");
      }

      for(int i=8;i>0;i--)
      {
        string filenum="../data/output/GLR_baboon_"+to_string(i)+".png";
        Gray_level_resolution(baboon_256,pic_trans,i);
        imwrite(filenum,pic_trans);
        namedWindow("Gray_level_resolution");
        imshow("Gray_level_resolution",pic_trans);
        printf("MSE of baboon Gray_level_resolution of %d bit is %f \n",i,MSE(baboon_256,pic_trans));
        printf("PSNR of lena Gray_level_resolution of %d bit is %f \n",i,PSNR(baboon_256,pic_trans));
        waitKey(0);
        destroyWindow("Gray_level_resolution");
      }     
    }
    else if(inputstring == "2-3lena")
    {
      bicubic(lena_256,pic256);
      bicubic(lena_256,pic128);
      bicubic(lena_256,pic64);
      bicubic(lena_256,pic32);
      for(int i=8;i>0;i--)
      {
        string filenum256="../data/output/lena_256_"+to_string(i)+".png";
        string filenum128="../data/output/lena_128_"+to_string(i)+".png";
        string filenum64="../data/output/lena_64_"+to_string(i)+".png";
        string filenum32="../data/output/lena_32_"+to_string(i)+".png";
        Gray_level_resolution(pic256,tran256,i);
        Gray_level_resolution(pic128,tran128,i);
        Gray_level_resolution(pic64,tran64,i);
        Gray_level_resolution(pic32,tran32,i);
        namedWindow("lena256",0);
        resizeWindow("lena256",256,256);
        namedWindow("lena128",0);
        resizeWindow("lena128",256,256);
        namedWindow("lena64",0);
        resizeWindow("lena64",256,256);
        namedWindow("lena32",0);
        resizeWindow("lena32",256,256);
        imwrite(filenum256,tran256);
        imwrite(filenum128,tran128);
        imwrite(filenum64,tran64);
        imwrite(filenum32,tran32);
        imshow("lena256",tran256);
        imshow("lena128",tran128);
        imshow("lena64",tran64);
        imshow("lena32",tran32);
        waitKey(0);
        destroyWindow("lena256");
        destroyWindow("lena128");
        destroyWindow("lena64");
        destroyWindow("lena32");
      }
    }
    else if(inputstring == "2-3baboon")
    {
      bicubic(baboon_256,pic256);
      bicubic(baboon_256,pic128);
      bicubic(baboon_256,pic64);
      bicubic(baboon_256,pic32);
      namedWindow("baboon256",0);
      resizeWindow("baboon256",256,256);
      namedWindow("baboon128",0);
      resizeWindow("baboon128",256,256);
      namedWindow("baboon64",0);
      resizeWindow("baboon64",256,256);
      namedWindow("baboon32",0);
      resizeWindow("baboon32",256,256);
      for(int i=8;i>0;i--)
      {
        string filenum256="../data/output/baboon_256_"+to_string(i)+".png";
        string filenum128="../data/output/baboon_128_"+to_string(i)+".png";
        string filenum64="../data/output/baboon_64_"+to_string(i)+".png";
        string filenum32="../data/output/baboon_32_"+to_string(i)+".png";
        Gray_level_resolution(pic256,tran256,i);
        Gray_level_resolution(pic128,tran128,i);
        Gray_level_resolution(pic64,tran64,i);
        Gray_level_resolution(pic32,tran32,i);
        imwrite(filenum256,tran256);
        imwrite(filenum128,tran128);
        imwrite(filenum64,tran64);
        imwrite(filenum32,tran32);
        imshow("baboon256",tran256);
        imshow("baboon128",tran128);
        imshow("baboon64",tran64);
        imshow("baboon32",tran32);
        waitKey(0);

      }
      destroyWindow("baboon256");
      destroyWindow("baboon128");
      destroyWindow("baboon64");
      destroyWindow("baboon32");
    }
    else if(inputstring == "2-5-1-d4")
    {
      unsigned char start_pt[2]={0,0},goal_pt[2]={19,19};
      unsigned char v1[3]={80,80,80};
      namedWindow("path",0);
      for(int i=0;i<=map.rows;i++)  //v[]setting
      {
        for(int j=0;j<map.cols;j++)
        {
          if(fmap.at<uchar>(i,j)==v1[0]||fmap.at<uchar>(i,j)==v1[1]||fmap.at<uchar>(i,j)==v1[2])
            map.at<uchar>(i,j)=1;
          else
            map.at<uchar>(i,j)=0;
          path.at<uchar>(i,j)=0;
        }
      }
      P4(0,0,map,path,-1);
      if(fin>=1)
        printf("pass,distance = %d \n",dis);
      else
        printf("fail\n");
      imshow("path",fullpath);
      imwrite("../data/output/D4path80.png",fullpath);
      waitKey(0);
      unsigned char saveraw[20*20];
      for(int i=0;i<map.rows;i++)
      {
        for(int j=0;j<map.cols;j++)
          saveraw[i*20+j]=fullpath.at<uchar>(i,j);
      }
      Write_Raw("../data/output/D4path80.raw",saveraw,20,20);
      destroyWindow("path");
    }
    else if(inputstring == "2-5-2-d4")
    {
      unsigned char start_pt[2]={0,0},goal_pt[2]={19,19};
      unsigned char v2[3]={80,160,80};
      namedWindow("path",0);
      for(int i=0;i<=map.rows;i++)  //v[]setting
      {
        for(int j=0;j<map.cols;j++)
        {
          if(fmap.at<uchar>(i,j)==v2[0]||fmap.at<uchar>(i,j)==v2[1]||fmap.at<uchar>(i,j)==v2[2])
            map.at<uchar>(i,j)=1;
          else
            map.at<uchar>(i,j)=0;
          path.at<uchar>(i,j)=0;
        }
      }
      P4(0,0,map,path,-1);
      if(fin>=1)
        printf("pass,distance = %d \n",dis);
      else
        printf("fail\n");
      imshow("path",fullpath);
      waitKey(0);
      imwrite("../data/output/D4path80_160.png",fullpath);
      unsigned char saveraw[20*20];
      for(int i=0;i<map.rows;i++)
      {
        for(int j=0;j<map.cols;j++)
          saveraw[i*20+j]=fullpath.at<uchar>(i,j);
      }
      Write_Raw("../data/output/D4path80_160.raw",saveraw,20,20);
      destroyWindow("path");      
    }
    else if(inputstring == "2-5-3-d4")
    {
      unsigned char start_pt[2]={0,0},goal_pt[2]={19,19};
      unsigned char v3[3]={80,160,255};
      namedWindow("path",0);
      for(int i=0;i<=map.rows;i++)  //v[]setting
      {
        for(int j=0;j<map.cols;j++)
        {
          if(fmap.at<uchar>(i,j)==v3[0]||fmap.at<uchar>(i,j)==v3[1]||fmap.at<uchar>(i,j)==v3[2])
            map.at<uchar>(i,j)=1;
          else
            map.at<uchar>(i,j)=0;
          path.at<uchar>(i,j)=0;
        }
      }
      P4(0,0,map,path,-1);
      if(fin>=1)
        printf("pass,distance = %d \n",dis);
      else
        printf("fail\n");
      imshow("path",fullpath);
      waitKey(0);
      imwrite("../data/output/D4path80_160_255.png",fullpath);
      unsigned char saveraw[20*20];
      for(int i=0;i<map.rows;i++)
      {
        for(int j=0;j<map.cols;j++)
          saveraw[i*20+j]=fullpath.at<uchar>(i,j);
      }
      Write_Raw("../data/output/D4path80_160_255.raw",saveraw,20,20);
      destroyWindow("path"); 
    }
    else if(inputstring == "2-5-1-d8")
    {
      unsigned char start_pt[2]={0,0},goal_pt[2]={19,19};
      unsigned char v1[3]={80,80,80};
      namedWindow("path",0);
      for(int i=0;i<=map.rows;i++)  //v[]setting
      {
        for(int j=0;j<map.cols;j++)
        {
          if(fmap.at<uchar>(i,j)==v1[0]||fmap.at<uchar>(i,j)==v1[1]||fmap.at<uchar>(i,j)==v1[2])
            map.at<uchar>(i,j)=1;
          else
            map.at<uchar>(i,j)=0;
          path.at<uchar>(i,j)=0;
        }
      }
      P8(0,0,map,path,-1);
      if(fin>=1)
        printf("pass,distance = %d \n",dis);
      else
        printf("fail\n");
      imshow("path",fullpath);
      waitKey(0);
      imwrite("../data/output/D8path80.png",fullpath);
      unsigned char saveraw[20*20];
      for(int i=0;i<map.rows;i++)
      {
        for(int j=0;j<map.cols;j++)
          saveraw[i*20+j]=fullpath.at<uchar>(i,j);
      }
      Write_Raw("../data/output/D8path80.raw",saveraw,20,20);
      destroyWindow("path");
    }
    else if(inputstring == "2-5-2-d8")
    {
      unsigned char start_pt[2]={0,0},goal_pt[2]={19,19};
      unsigned char v2[3]={80,160,80};
      namedWindow("path",0);
      for(int i=0;i<=map.rows;i++)  //v[]setting
      {
        for(int j=0;j<map.cols;j++)
        {
          if(fmap.at<uchar>(i,j)==v2[0]||fmap.at<uchar>(i,j)==v2[1]||fmap.at<uchar>(i,j)==v2[2])
            map.at<uchar>(i,j)=1;
          else
            map.at<uchar>(i,j)=0;
          path.at<uchar>(i,j)=0;
        }
      }
      P8(0,0,map,path,-1);
      if(fin>=1)
        printf("pass,distance = %d \n",dis);
      else
        printf("fail\n");
      imshow("path",fullpath);
      waitKey(0);
      imwrite("../data/output/D8path80_160.png",fullpath);
      unsigned char saveraw[20*20];
      for(int i=0;i<map.rows;i++)
      {
        for(int j=0;j<map.cols;j++)
          saveraw[i*20+j]=fullpath.at<uchar>(i,j);
      }
      Write_Raw("../data/output/D8path80_160.raw",saveraw,20,20);
      destroyWindow("path");
    }
    else if(inputstring == "2-5-3-d8")
    {
        
    }
  }

  return 0;
}
