#include <iostream>
#include <cmath>
#include <map>
#include <cstdlib>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>
#include <utility>

using namespace std;
using namespace Eigen;

typedef long double scalar;
typedef Matrix<scalar, Dynamic, Dynamic> mat;
typedef pair<mat /* problem matrix */, mat /* rhs, 'sign count' */> lpsat;

const scalar one = 1;
const scalar lambda = 0;
scalar lastloss;
const scalar pi = acos(-one);
uint iter = 0, i2pr = 250;
mat M, sinx2, sin2x, cos2x, H, g, mr;

void newtonupdate(const lpsat& lp, mat& x) { 
	mr = mat::Identity(lp.second.rows(), lp.second.rows());
	for (uint n = 0; n < lp.second.rows(); n++) mr(n,n) = lp.second(n, 0) ? one / lp.second(n, 0) : 0;
	M = mr * lp.first;
	sinx2 = x.array().sin().square();
	sin2x = (x*2.).array().sin();
	cos2x = (x*2.).array().cos();
	H = M * cos2x.asDiagonal() * 2.;
	g = sin2x.transpose() * M.transpose();
        JacobiSVD<mat> svd(H, ComputeThinU | ComputeThinV);
	x += svd.solve(g.transpose());
	if (iter % i2pr == 0) {
		cout<<"min err:"<<endl<<(lp.first * sinx2 - lp.second).maxCoeff()<<endl;
		cout<<"grad norm:"<<endl<<g.norm()<<endl;
		cout<<"grad:"<<endl<<g<<endl;
		cout<<"Hessian singular vals:"<<endl<<svd.singularValues().transpose()<<endl;
	        cout<<"xh:"<<endl<<x.transpose()<<endl;
	        cout<<"sin(xh)^2:"<<endl<<sinx2.transpose()<<endl<<endl<<iter<<endl;
	}
}

lpsat dimacs2eigen(istream& is) {
	string str;
	uint rows, cols;
	mat m, rhs;

	do { getline(is, str);	} while (str[0] == 'c');
	sscanf(str.c_str(), "p cnf %d %d", &cols, &rows);
	m = mat::Zero(rows + 2 * cols, cols);
	rhs = mat(rows + 2 * cols, 1);
	uint n = 0;
	for (; n < rows; n++) {
		getline(is, str);
		int v1, v2, v3;
		sscanf(str.c_str(), "%d %d %d", &v1, &v2, &v3);
		m(n,abs(v1) - 1) = v1 > 0 ? -1 : 1;
		m(n,abs(v2) - 1) = v2 > 0 ? -1 : 1;
		m(n,abs(v3) - 1) = v3 > 0 ? -1 : 1;
		rhs(n,0) = (v1 > 0 ? 0 : 1) + (v2 > 0 ? 0 : 1) + (v3 > 0 ? 0 : 1) - 1;
	}
	for (; n < rows + cols; n++) {
		m(n, n - rows) = 1; // x<=1
		rhs(n, 0) = -3;
	}
        for (; n < rows + 2 * cols; n++) {
                m(n, n - rows - cols) = -1; // -x<=1
                rhs(n, 0) = -3;
        }

//	mat a(1, cols); for (uint n=1;n<=cols;n++)a(0,n-1)=n;
//	cout << a << endl << m << endl << a << endl << rhs.transpose() << endl;
	return lpsat(m, rhs);
}

int main(int argc,char** argv){
	lpsat p = dimacs2eigen(cin);
//	cout<<p.first<<endl;
	JacobiSVD<mat> svd(p.first, ComputeFullU | ComputeFullV);
	mat xh = svd.solve(p.second), x = mat::Ones(p.first.cols(), 1) * 3;
	cout << endl << "D:" << endl << svd.singularValues().transpose() << endl
//		<< endl << "U:" << endl << svd.matrixU().row(1) << endl
//		<< endl << "U:" << endl << svd.matrixU().col(1).transpose() << endl
//		<< endl << "V:" << endl << svd.matrixV().row(1) << endl
//		<< endl << "V:" << endl << svd.matrixV().col(1).transpose() << endl
		<< endl << "V:" << endl << svd.matrixV() << endl
		<< endl << "det:" << endl << svd.singularValues().prod() << endl
		<< endl << "det^2:" << endl << pow(svd.singularValues().prod(),2) << endl
		<< endl << "xh:" << endl << xh.norm() << endl << xh.mean() << endl;

//	for (iter = 0;iter < 1000000; iter++)  
//		newtonupdate(p, x); 

        return 0;
}
