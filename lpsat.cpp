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

const scalar lambda = 0;
scalar lastloss;
mat lgup;
uint iter = 0;
mat h(const lpsat& lp, const mat& x) { 
	mat r = lp.first * mat(x.array().sin().square()); 
	return r;
}

mat gh(const lpsat& lp, const mat& x) { 
	mat r = (lp.first * mat((x*2.).array().sin()).asDiagonal()).transpose();	
        return r;
}

scalar loss(const lpsat& lp, const mat& x) { 
	lastloss = (h(lp, x) /*- lp.second*/).squaredNorm() * .5;
        if (iter % 10000 == 0) cout<<"loss:"<<endl<<(sqrt(lastloss)/p.first.rows())<<endl; 
        return lastloss;
}

mat gloss(const lpsat& lp, const mat& x) {
        return gh(lp, x) * (h(lp, x)/* - lp.second*/);
}

void gdupdate(const lpsat& lp, mat& x) { 
	lgup = gloss(lp, x);
	x += lgup;
	if (iter % 10000 == 0) {
	        cout<<"xh:"<<endl<<x.transpose()<<endl;
	        cout<<"s2xh:"<<endl<<x.array().sin().square().transpose()<<endl;
	}
}

lpsat dimacs2eigen(istream& is) {
	string str;
	uint rows, cols;
	mat m, rhs;

	do { getline(is, str);	} while (str[0] == 'c');
	sscanf(str.c_str(), "p cnf %d %d", &cols, &rows);
	m = mat::Zero(rows + cols, cols);
	rhs = mat(rows + cols, 1);

	for (uint n = 0; n < rows; n++) {
		getline(is, str);
		int v1, v2, v3;
		sscanf(str.c_str(), "%d %d %d", &v1, &v2, &v3);
		m(n,abs(v1) - 1) = v1 > 0 ? -1 : 1;
		m(n,abs(v2) - 1) = v2 > 0 ? -1 : 1;
		m(n,abs(v3) - 1) = v3 > 0 ? -1 : 1;
		rhs(n,0) = (v1 > 0 ? 0 : 1) + (v2 > 0 ? 0 : 1) + (v3 > 0 ? 0 : 1) - 1;
	}
	for (uint n = rows; n < rows + cols; n++) {
		rhs(n, 0) = 0;
		m(n, n - rows) = sqrt(lambda);
	}

//	mat a(1, cols); for (uint n=1;n<=cols;n++)a(0,n-1)=n;
//	cout << a << endl << m << endl << a << endl << rhs.transpose() << endl;
	return lpsat(m, rhs);
}

int main(int argc,char** argv){
	lpsat p = dimacs2eigen(cin);
	JacobiSVD<mat> svd(p.first, ComputeThinU | ComputeThinV);
	mat xh = svd.solve(p.second), x = mat::Ones(p.first.cols(), 1) * .5;
//	cout<<p.first<<endl<<p.second<<endl;
	cout << endl << "xh:" << endl << xh.norm() << endl << xh.mean() << endl;

	for (iter = 0;iter < 1000000; iter++) { 
		if (iter % 10000 == 0) cout<<loss(p, x)<<','; 
		gdupdate(p, x); 
	}

        return 0;
}
