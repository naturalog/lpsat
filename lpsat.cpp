#include <iostream>
#include <cmath>
#include <map>
#include <cstdlib>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>
#include <utility>
#include <mpreal.h>
#include <eigen3/unsupported/Eigen/MPRealSupport>
#include <algorithm>

using namespace mpfr;
using namespace std;
using namespace Eigen;

typedef mpreal scalar;
typedef Matrix<scalar, Dynamic, Dynamic> mat;
typedef pair<mat /* problem matrix */, mat /* rhs, 'sign count' */> lpsat;

const scalar one = 1;

/*
const scalar lambda = 0;
scalar lastloss;
const scalar pi = acos(-one);
uint iter = 0, i2pr = 250;
mat M, sinx2, sin2x, cos2x, H, g, mr;

void newtonupdate(const lpsat& lp, mat& x) { 
	M = lp.first;
	mat MTM = M.transpose() * M;
	mat MTy = M.transpose() * lp.second;
	mat r = MTM * x - MTy;
	mat g = MTM.transpose() * r;
	mat H = MTM.transpose() * MTM;
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
*/

const bool constrain = false;

#include <vector>

mat unit(int n, uint d) {
	mat r;
	if (n < 0) {
		r = mat::Identity(d, d);
		r(-1-n, -1-n) = 0;
		return r;
	}
	r = mat::Zero(d, d);
	r(n-1, n-1) = 1;
	return r;
}

void dimacs2projection(istream& is) {
        string str;
        uint rows, cols;
	int v1, v2, v3;

        do { getline(is, str);  } while (str[0] == 'c');
        sscanf(str.c_str(), "p cnf %d %d", &cols, &rows);
	mat t, r, I = mat::Identity(cols, cols);
	r = I;

        for (uint n = 0; n < rows; n++) {
                getline(is, str);
                sscanf(str.c_str(), "%d %d %d", &v1, &v2, &v3);
		t = (unit(v1, cols) + unit(v2, cols) + unit(v3, cols));
		r *= t;
		cout<<"t:"<<endl<<t.diagonal().transpose()<<endl<<"r:"<<endl<<r.diagonal().transpose()<<endl;
        }
}


lpsat dimacs2eigen(istream& is) {
	string str;
	uint rows, cols;
	mat m, rhs;

	do { getline(is, str);	} while (str[0] == 'c');
	sscanf(str.c_str(), "p cnf %d %d", &cols, &rows);
	m = mat::Zero(rows + (constrain ? 2 * cols : 0), cols);
	rhs = mat(rows + (constrain ? 2 * cols : 0), 1);
	uint n = 0;
	for (; n < rows; n++) {
		getline(is, str);
		int v1, v2, v3;
		sscanf(str.c_str(), "%d %d %d", &v1, &v2, &v3);
		m(n,abs(v1) - 1) = v1 > 0 ? 1 : -1;
		m(n,abs(v2) - 1) = v2 > 0 ? 1 : -1;
		m(n,abs(v3) - 1) = v3 > 0 ? 1 : -1;
		rhs(n,0) = -2;//(v1 > 0 ? 0 : 1) + (v2 > 0 ? 0 : 1) + (v3 > 0 ? 0 : 1) - 1;
	}
	if (constrain) {
		const scalar lambda = 0*2;//sqrt(one*.5);
		for (; n < rows + cols; n++) {
			// x >= -1
			m(n, n - rows) = lambda;
			rhs(n, 0) = -lambda;
		}
	        for (; n < rows + 2 * cols; n++) {
			// -x >= -1
	                m(n, n - rows - cols) = -lambda;
	                rhs(n, 0) = -lambda;
	        }
	}

//	mat a(1, cols); for (uint n=1;n<=cols;n++)a(0,n-1)=n;
//	cout << a << endl << m << endl << a << endl << rhs.transpose() << endl;
	return lpsat(m, rhs);
}

void symbolic(lpsat lp) {
	 for (uint r = 0; r < lp.first.rows(); r++) {
	 	for (uint c = 0; c < lp.first.cols(); c++)
			if (lp.first(r, c))
				cout << (lp.first(r, c) > 0 ? '+' : '-') << "x[" << c << "] ";
		cout << "+ " << -lp.second(r, 0) << " >= 0;" << endl;
	}
}

void prod_symbolic(lpsat lp) {
         for (uint r = 0; r < lp.first.rows(); r++) {
		cout<<'(';
                for (uint c = 0; c < lp.first.cols(); c++)
                        if (lp.first(r, c))
                                cout << (lp.first(r, c) > 0 ? '+' : '-') << "x[" << c << "] ";
                cout << "+ " << -lp.second(r, 0) << ") *";
        }
	cout<<endl;
}

mat hinge(mat x, mat y, uint upto) {
	upto = x.rows();
	mat r(upto, 1);
	for (uint n = 0; n < upto; n++) 
		r(n, 0) = std::max<scalar>(y(n, 0) - x(n, 0), 0);
	return r;
}

int main(int argc,char** argv){
	mpreal::set_default_prec(4096);
	dimacs2projection(cin);
	return 0;
	lpsat p = dimacs2eigen(cin);
//	prod_symbolic(p); return 0;
	scalar a,b;
	cout<<p.first<<endl;
	JacobiSVD<mat> svd(p.first, ComputeFullU | ComputeFullV);
	mat xh = svd.solve(p.second), x = mat::Ones(p.first.cols(), 1);
	cout << endl << "MTM:" << endl << p.first.transpose() * p.first << endl
		<< endl << "MMT:" << endl << p.first * p.first.transpose() << endl
		<< endl << "D^2:" << endl << svd.singularValues().array().square().transpose() << endl
                << endl << "desired norm of input vector: " << x.norm()
                << endl << "desired norm of output vector: " << (p.first.transpose() * p.second).norm()
		<< endl << "ratio: " << (p.first.transpose() * p.second).norm() / x.norm() 
		<< endl << "D1/Dn: " << svd.singularValues()(1)/svd.singularValues()(svd.singularValues().size()-1) 
//		<< endl << "U:" << endl << svd.matrixU().row(1) << endl
//		<< endl << "U:" << endl << svd.matrixU().col(1).transpose() << endl
//		<< endl << "V:" << endl << svd.matrixV().row(1) << endl
//		<< endl << "V:" << endl << svd.matrixV().col(1).transpose() << endl
		<< endl << "U:" << endl << svd.matrixU() << endl
		<< endl << "V:" << endl << svd.matrixV() << endl
		<< endl << "sqrt(det):" << endl << svd.singularValues().prod() << endl
		<< endl << "det:" << endl << pow(svd.singularValues().prod(),2) << endl
		<< endl << "xh:" << xh.transpose() << endl << xh.norm() << endl << xh.mean() << endl
		<< endl << "Mxh:" << endl << (p.first * xh).transpose() << endl
		<< endl << "rhs:" << endl << p.second.transpose() << endl
		<< endl << "E(T=>0) : " << endl << (p.first * xh - p.second).transpose() << endl
		<< endl << "hinge:" << endl << hinge(p.first * xh, p.second, p.first.cols()).transpose() << endl
		<< endl << "||hinge||:" << endl << -hinge(p.first * xh, p.second, p.first.cols()).sum() << endl
		<< endl << "1M:" << endl << (mat::Ones(1, p.first.rows()) * p.first).cwiseAbs() << endl
		<< endl << "M(1/2):" << endl << p.first * mat::Ones(p.first.rows(), 1)/2 << endl
		<< endl << "1M(1/2):" << endl << mat::Ones(1, p.first.rows()) * p.first * mat::Ones(p.first.rows(), 1) / 2 << endl;

//	for (iter = 0;iter < 1000000; iter++)  
//		newtonupdate(p, x); 

        return 0;
}
