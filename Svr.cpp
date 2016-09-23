#include <iostream>
#include <string.h>
#include <fstream>
#include <algorithm>
#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#define N 140//样本总数
#define end_support_i 120  //140条训练集
#define Dimen 5 //维数
#define INF 0x7f7f7f7f
using namespace std;
const int C=1024;    //惩罚因子
const int first_test_i=81;
const double eps=1e-3;//一个近似0的小数
const int two_sigma_squared=2;//RBF(Radial-Basis Function)核函数中的参数
double alph[2*end_support_i];//alph扩大为2倍
int    y[2*end_support_i];//存放结果值yi
double G[2*end_support_i];
double b;//偏置
double p=0.125;//eplison参数
double tau=1e-12;
double target[N];//训练与测试样本的目标值
double dense_points[N][Dimen];//存放训练与测试样本，0-end_support_i-1训练;first_test_i-N测试
//函数的申明
int Ansi=0,Ansj=0;
int takeStep(int,int);
double TestStudy(int k);
double kernel_funcp(int,int);
double kernel_func(int,int);
double dot_product_func(int,int);
void  OutPutFile();
void SelectAns()//选取一个需要进行优化的数据集
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
void setX() //录入样本数据
{
	ifstream inClientFile("battery_data.txt", ios::in);//ifstream用于从指定文件输入
	if(!inClientFile)
	{
		cerr<<"File could not be opened!"<<endl;
		exit(1);//exit的作用为终止程序。
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
	inClientFile.close();//显式关闭不再引用的文件。
/*	for(i=0;i<20;i++)
	{
		cout<<target[i]<<endl;
	}*/
}
void Initialize()
{
	b=0.0;//初始化偏置b 0
	memset(alph,0,sizeof(alph));//初始化alph向量
    memset(y,-1,sizeof(y));
	setX();//录入数据
}
int main()
{
Initialize();
int iter=0;
while(1)
{
	SelectAns();//选出两个变量作为待调整变量
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
	if(iter++>1985900)
		break;
	if(iter%1000==0)
	  cout<<"Iter: "<<iter<<endl;
	}
	cout<<"Iter: "<<iter<<endl;
	for(int i=0;i<140 ;i++)//这是输出预测的东西了
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
		outClientFile<<"rate="<<(double)n_support_vectors/first_test_i<<endl;
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
	s-=b;   //因为上面计算中  b=-b
	return s;
}
double dot_product_func(int i1,int i2)//计算向量内积
{
	double dot=0;
	for(int i=0;i<Dimen;i++)
	{
		dot+=dense_points[i1][i]*dense_points[i2][i];
	}
	return dot;
}
//The kernel_func(int, int) is RBF(Radial-Basis Function).
//K(Xi, Xj)=exp(-||Xi-Xj||^2/(r))   //r的值需要人为指定
double kernel_func(int i1,int i2)//处理后的径向基核函数  YiYjKernel(i,j)其实也可以不出理
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
double kernel_funcp(int i1,int i2)//预测中使用径向基核函数使用
{
    double Ans=0;
	double s=dot_product_func(i1,i2);
	s*=-2;
	s+=dot_product_func(i1,i1)+dot_product_func(i2,i2);
	Ans=-exp(-s/two_sigma_squared);
	return Ans;
}
