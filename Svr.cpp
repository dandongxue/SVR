#include <iostream>
#include <string.h>
#include <fstream>
#include <algorithm>
#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#define N 140              //Total Sample Num
#define end_support_i 140  //Training Sample Num
#define Dimen 5 //The Sample Dimension
#define INF 0x7f7f7f7f
using namespace std;
const int C=1024;    //Paramter C
const int first_test_i=1;
const double eps=1e-3;
const int two_sigma_squared=2;//RBF(Radial-Basis Function)Parameter
double alph[2*end_support_i];//alph to be double
int    y[2*end_support_i];//Get The Ans Y
double G[2*end_support_i];
double b;//thresh b
double p=0.125;//Eplison Parameter
double tau=1e-12;
double target[N];//Store All The Samples Target
double dense_points[N][Dimen];//Store Samles，0-end_support_i-1 For Training;first_test_i-N For Testing
int Ansi=0,Ansj=0;
int takeStep(int,int);
double TestStudy(int k);
double kernel_funcp(int,int);
double kernel_func(int,int);
double dot_product_func(int,int);
void  OutPutFile();
void SelectAns()//To choose one datasets which needs to be optimization
{
Ansi=Ansj=-1;
double G_max=-INF,G_max2=-INF;
double G_min=INF;
for(int i=0;i<2*end_support_i;i++)
{
	if(y[i]==1)
	{
	  if(alph[i]<C)
		  if(-G[i]>=G_max)
		  {
		  Ansi=i;
		  G_max=-G[i];
		  }
	}
	else
	{
	if(alph[i]>0)
		if(G[i]>=G_max)
		{
		G_max=G[i];
		Ansi=i;
		}
	}
}
double obj_min=INF;
for(int i=0;i<2*end_support_i;i++)
{
  if((y[i]==1&&alph[i]>0)||(y[i]==-1)&&alph[i]<C)
  {
	  b=G_max+y[i]*G[i];//Ansi已经找到 求Bij可以直接y[i]*G[i]+G_max
	  if(-y[i]*G[i]<=G_min)
		  G_min=-y[i]*G[i];
	  if(b>0)
	  {
		double a=kernel_func(Ansi,Ansi)+kernel_func(i,i)-2*kernel_func(Ansi,i);
		if(a<=0)
			a=tau;
		if((-(b*b)/a) <=obj_min)
		{
		Ansj=i;
		obj_min=-(b*b)/a;
		}
	  }
  }
}
if(G_max-G_min<eps)
{
	 Ansi=-1;
	 Ansj=-1;
}
}
void setX() //Reading Data
{
	ifstream inClientFile("battery_data.txt", ios::in);//ifstream用于从指定文件输入
	if(!inClientFile)
	{
		cerr<<"File could not be opened!"<<endl;
		exit(1);//Exit To End
	}
	int i=0,j=0;
	double a_data;//a_data为每次读到的数据, 默认为6位有效数字。
	while(inClientFile>>a_data)
	{
		if(j==Dimen-1)
		{
			target[i]=a_data;
			y[i]=1;
			G[i+end_support_i]=-(p+target[i]);
			G[i]=target[i]-p;
			j=0;
			i++;
		}
		else{
			dense_points[i][j]=a_data;//录入数据到dense_points中  包括训练集和测试集
			j++;
			}	
	}
	inClientFile.close();
}
void Initialize()
{
	b=0.0;//Init b
	memset(alph,0,sizeof(alph));
        memset(y,-1,sizeof(y));
	setX();//Reading Data
}
int main()
{
Initialize();
int iter=0;
while(1)
{
	SelectAns();//Select Two Variable To Adjust
	if(Ansj==-1)
	   break;
	double a,OldAi,OldAj,deltaAi,deltaAj,sum;
        a=kernel_func(Ansi,Ansi)+kernel_func(Ansj,Ansj)-2*kernel_func(Ansi,Ansj);
	if(a<=0)a=tau;
	OldAi=alph[Ansi],OldAj=alph[Ansj];
	if(y[Ansi]!=y[Ansj])
	{
	  double delta=(-G[Ansi]-G[Ansj])/a;
	  double diff=alph[Ansi]-alph[Ansj];
	  alph[Ansi]+=delta;
	  alph[Ansj]+=delta;
	  //将数据限定在合理的数据区域
	  if(diff>0)
	  {
	    if(alph[Ansj]<0)
		{
		alph[Ansj]=0;
		alph[Ansi]=diff;
		}
	  }
	  else 
	  {
	    if(alph[Ansi]<0)
		{
		alph[Ansi]=0;
		alph[Ansj]=-diff;
		}
	  }
	  if(diff>0)
	  {
	    if(alph[Ansi]>C)
		{
		alph[Ansi]=C;
		alph[Ansj]=C-diff;
		}
	  }
	  else
	  {
		if(alph[Ansj]>C)
		{
		alph[Ansj]=C;
		alph[Ansi]=C+diff;
		}
	  }
	}
	else
	{
	  double delta=(G[Ansi]-G[Ansj])/a;
	  double sum=alph[Ansi]+alph[Ansj];
	  alph[Ansi]-=delta;
	  alph[Ansj]+=delta;
	  //将数据限定在合理的数据区域
	  if(sum > C)
			{
				if(alph[Ansi] > C)
				{
					alph[Ansi] = C;
					alph[Ansj] = sum - C;
				}
			}
			else
			{
				if(alph[Ansj] < 0)
				{
					alph[Ansj] = 0;
					alph[Ansi] = sum;
				}
			}
			if(sum > C)
			{
				if(alph[Ansj] > C)
				{
					alph[Ansj] = C;
					alph[Ansi] = sum - C;
				}
			}
			else
			{
				if(alph[Ansi] < 0)
				{
					alph[Ansi] = 0;
					alph[Ansj] = sum;
				}
			}
	}
	deltaAi=alph[Ansi]-OldAi,deltaAj=alph[Ansj]-OldAj;
	for(int i=0;i<2*end_support_i;i++)
	  G[i]+=kernel_func(i,Ansi)*deltaAi+kernel_func(i,Ansj)*deltaAj;
	if(iter++>2985900)//The Bigest Iter Num!
		break;
	if(iter%3000==0)
	  cout<<"Iter:[ "<<iter<<" ]"<<endl;
	}
	cout<<"Iter: "<<iter<<endl;
	for(int i=0;i<N;i++)//这是输出预测的东西了
		cout<<"Precate: "<<TestStudy(i)<<"  True:"<<target[i]<<"  Dis:"<<fabs(TestStudy(i)-target[i])<<endl;		
	OutPutFile();
	return 0;
}
void  OutPutFile()
{
	ofstream outClientFile("data_result.txt", ios::out);
	outClientFile<<"Dimension="<<Dimen<<endl;//维数
	outClientFile<<"b="<<b<<endl;//threshold
	outClientFile<<"two_sigma_squared="<<two_sigma_squared<<endl;
    	outClientFile<<"C="<<C<<endl;
	int n_support_vectors=0;
		for(int i=0;i<end_support_i;i++)
		{
		    if(alph[i]>0&&alph[i]<C)
			{
				n_support_vectors++;
			}
		}
		outClientFile<<"n_support_vectors="<<n_support_vectors<<endl;
		outClientFile<<"support vector rate="<<(double)n_support_vectors/end_support_i<<endl;
		for(int i=0;i<end_support_i;i++)
		{
				outClientFile<<"alph["<<i<<"]="<<alph[i]-alph[i+end_support_i]<<endl;
		}
		outClientFile<<endl;
}
double TestStudy(int k)
{
    double s=0.0;
	for(int i=0;i<end_support_i;i++)
	{
			s+=(alph[i]-alph[i+end_support_i])*kernel_funcp(i,k);		
	}
	s-=b;   //Because  b=-b
	return s;
}
double dot_product_func(int i1,int i2)//Dot product
{
	double dot=0;
	for(int i=0;i<Dimen;i++)
	{
		dot+=dense_points[i1][i]*dense_points[i2][i];
	}
	return dot;
}
//The kernel_func(int, int) is RBF(Radial-Basis Function).
//K(Xi, Xj)=exp(-||Xi-Xj||^2/(r))   
double kernel_func(int i1,int i2)//The New RBF Kenel 
{
	double Ans=0;
	if(i1<end_support_i&&i2<end_support_i)
	{
		double s=dot_product_func(i1,i2);
		s*=-2;
		s+=dot_product_func(i1,i1)+dot_product_func(i2,i2);
		Ans=exp(-s/two_sigma_squared);
		return Ans;
	}
	else if(i1>=end_support_i&&i2>=end_support_i)
	{   i1-=end_support_i;i2-=end_support_i;
		double s=dot_product_func(i1,i2);
		s*=-2;
		s+=dot_product_func(i1,i1)+dot_product_func(i2,i2);
		Ans=exp(-s/two_sigma_squared);
		return Ans;
	}
	else if(i1>=end_support_i||i2>=end_support_i)
	{
		i1=i1>=end_support_i?(i1-end_support_i):i1;
		i2=i2>=end_support_i?(i2-end_support_i):i2;
		double s=dot_product_func(i1,i2);
		s*=-2;
		s+=dot_product_func(i1,i1)+dot_product_func(i2,i2);
		Ans=-exp(-s/two_sigma_squared);
		return Ans;
	}
	return Ans;
}
double kernel_funcp(int i1,int i2)//Kenel Of RBF
{
    	double Ans=0;
	double s=dot_product_func(i1,i2);
	s*=-2;
	s+=dot_product_func(i1,i1)+dot_product_func(i2,i2);
	Ans=-exp(-s/two_sigma_squared);
	return Ans;
}
