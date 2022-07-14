// #include "stdafx.h"
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

using namespace Eigen;
using namespace std;
int main()
{
  // A为自己构造的输入判别矩阵
  MatrixXd A(4,4); //MatrixXd类型，代表一个任意大小的矩阵
  A << 1,     3,   1,    0.333,
       0.333, 1,   0.5,  0.2,
       1,     2,   1,    0.333,
       3,     5,   3,    1;
  cout << A << endl;
  // 查看行数和列数
  int m=A.rows();
  int n=A.cols();
  cout <<"m:"<< m << endl;
  cout <<"n:"<< n << endl;
  // 求特征值和特征向量
  EigenSolver<Matrix<double, 4, 4>> es(A);
  MatrixXcd evecs = es.eigenvectors();//获取矩阵特征向量4*4，这里定义的MatrixXcd必须有c，表示获得的是complex复数矩阵
  MatrixXcd evals = es.eigenvalues();//获取矩阵特征值 4*1
  MatrixXd evalsReal;//注意这里定义的MatrixXd里没有c
  evalsReal=evals.real();//获取特征值实数部分
  MatrixXf::Index evalsMax;
  evalsReal.rowwise().sum().maxCoeff(&evalsMax);//得到最大特征值的位置
  Vector4d q;
  q << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax), evecs.real()(2, evalsMax), evecs.real()(3, evalsMax);//得到对应特征向量
  // 赋权重
  MatrixXd  weight;
  weight= MatrixXd::Zero(n, 1);
  double sum = 0;
  for(int i =0; i<n; i++)
  {
    sum = sum+q(i);
  }
  for(int i =0; i<n; i++)
  {
    weight(i)= q(i)/sum;
  }
  cout <<"evalsMax:"<< evalsReal(evalsMax) << endl;
  cout <<"q:"<< q << endl;
  cout <<"weight:"<< weight << endl;

  // 一致性检验
  double RI[15]={0.0,0.0,0.58,0.9,1.12,1.24,1.32,1.41,1.45,1.49,1.52,1.54,1.56,1.58,1.59};
  
  double CI = (evalsReal(evalsMax)-n)/(n-1);
  
  double CR = CI/RI[n-1];
  if(CR>=0.1)
  {
    cout<<"没有通过一致性检测\n"<<endl<<endl;
  }
  else
  {
    cout<<"通过一致性检验\n"<<endl<<endl;
  }
  return 0;
}


