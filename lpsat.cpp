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
mat lgup;
uint iter = 0, i2pr = 1000;

//// hypothesis
//mat h(const lpsat& lp, const mat& x)	{ return lp.first * mat(x.array().sin().square()); }
////gradient
//mat gh(const lpsat& lp, const mat& x)	{ return (lp.first * mat((x*2.).array().sin()).transpose()).transpose(); }
////hessian
//mat Hh(const lpsat& lp, const mat& x)	{ return (lp.first * mat((x*2.).array().cos()).asDiagonal()) * 2.; }
//
//scalar loss(const lpsat& lp, const mat& x) { 
//	lastloss = (h(lp, x) /*- lp.second*/).squaredNorm() * .5 - lambda * x.squaredNorm();
//        if (iter % i2pr == 0) cout<<"loss:"<<endl<<(sqrt(lastloss)/(lp.first.rows()/*-lp.first.cols()*/))<<endl; 
//        return lastloss;
//}
//
//mat gloss(const lpsat& lp, const mat& x) {
//        return gh(lp, x) * (h(lp, x)/* - lp.second*/) - lambda * x;
//}
//
//mat Hloss(const lpsat& lp, const mat& x) {
//        mat outer, r = Hh(lp, x) * h(lp, x) + (outer=gh(lp, x).transpose() * gh(lp, x));
//	if (outer.rows() == 1) throw 0;
//	return r;
//}

void gdupdate(const lpsat& lp, mat& x) { 
//	lgup = gloss(lp, x);
	mat M = lp.first;
	mat sin2x = (x*2.).array().sin();
	mat cos2x = (x*2.).array().cos();
	mat H = M * cos2x.asDiagonal() * 2.;
	mat g = sin2x.transpose() * M.transpose();
        JacobiSVD<mat> svd(H, ComputeThinU | ComputeThinV);
	x += svd.solve(g.transpose());
	if (iter % i2pr == 0) {
		cout<<"grad norm:"<<endl<<g.norm()<<endl;
		cout<<"grad:"<<endl<<g<<endl;
		cout<<"Hessian singular vals:"<<endl<<svd.singularValues().transpose()<<endl;
	        cout<<"xh:"<<endl<<x.transpose()<<endl;
	        cout<<"sin(xh)^2:"<<endl<<x.array().sin().square().transpose()<<endl<<endl<<iter<<endl;
	}
}

lpsat dimacs2eigen(istream& is) {
	string str;
	uint rows, cols;
	mat m, rhs;

	do { getline(is, str);	} while (str[0] == 'c');
	sscanf(str.c_str(), "p cnf %d %d", &cols, &rows);
	m = mat::Zero(rows/* + cols*/, cols);
	rhs = mat(rows/* + cols*/, 1);

	for (uint n = 0; n < rows; n++) {
		getline(is, str);
		int v1, v2, v3;
		sscanf(str.c_str(), "%d %d %d", &v1, &v2, &v3);
		m(n,abs(v1) - 1) = v1 > 0 ? -1 : 1;
		m(n,abs(v2) - 1) = v2 > 0 ? -1 : 1;
		m(n,abs(v3) - 1) = v3 > 0 ? -1 : 1;
		rhs(n,0) = (v1 > 0 ? 0 : 1) + (v2 > 0 ? 0 : 1) + (v3 > 0 ? 0 : 1) - 1;
	}
//	for (uint n = rows; n < rows + cols; n++) {
//		rhs(n, 0) = 0;
//		m(n, n - rows) = sqrt(lambda);
//	}

//	mat a(1, cols); for (uint n=1;n<=cols;n++)a(0,n-1)=n;
//	cout << a << endl << m << endl << a << endl << rhs.transpose() << endl;
	return lpsat(m, rhs);
}

int main(int argc,char** argv){
	lpsat p = dimacs2eigen(cin);
	JacobiSVD<mat> svd(p.first, ComputeThinU | ComputeThinV);
	mat xh = svd.solve(p.second), x = mat::Ones(p.first.cols(), 1) * pi * .125;
//	cout<<p.first<<endl<<p.second<<endl;
	cout << endl << "xh:" << endl << xh.norm() << endl << xh.mean() << endl;

	for (iter = 0;iter < 100000000; iter++) { 
//		if (iter % i2pr == 0) cout<<endl<<loss(p, x)<<endl; 
		gdupdate(p, x); 
	}

        return 0;
}
