#include <iostream>
#include <cmath>
#include <map>
#include <cstdlib>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>
#include <utility>

using namespace std;
using namespace Eigen;

typedef long double scalar;
typedef Matrix<scalar, Dynamic, Dynamic> mat;
typedef pair<mat /* problem matrix */, mat /* rhs, 'sign count' */> lpsat;

lpsat dimacs2eigen(istream& is) {
	string str;
	uint rows, cols;
	mat m, rhs;

	do { getline(is, str);	} while (str[0] == 'c');
	sscanf(str.c_str(), "p cnf %d %d", &cols, &rows);
	m = mat::Zero(rows, cols);
	rhs = mat(rows, 1);

	for (uint n = 0; n < rows; n++) {
		getline(is, str);
		int v1, v2, v3;
		sscanf(str.c_str(), "%d %d %d", &v1, &v2, &v3);
		m(n,abs(v1) - 1) = v1 > 0 ? -1 : 1;
		m(n,abs(v2) - 1) = v2 > 0 ? -1 : 1;
		m(n,abs(v3) - 1) = v3 > 0 ? -1 : 1;
		rhs(n,0) = (v1 > 0 ? 0 : 1) + (v2 > 0 ? 0 : 1) + (v3 > 0 ? 0 : 1) - 1;
	}
	mat a(1, cols); for (uint n=1;n<=cols;n++)a(0,n-1)=n;
//	cout << a << endl << m << endl << a << endl << rhs.transpose() << endl;
	return lpsat(m, rhs);
}

//pair<scalar, mat> objective(const mat& x, const lpsat& lp) {
//	scalar r = (cnf * x).squaredNorm();
//	return make_pair(r, cnf * r);
//}

int main(int argc,char** argv){
	lpsat p = dimacs2eigen(cin);
	JacobiSVD<mat> svd(p.first, ComputeThinU | ComputeThinV);
//	cout << "D:" << endl << svd.singularValues().transpose() << endl;
//	cout<<j.squaredNorm()<<endl;
	mat xh = svd.solve(p.second);
	cout << endl << "xh:" << endl << xh.norm() << endl << xh.mean() << endl;
//	cout << endl << (p.second.transpose()-svd.solve(p.second)).norm() << endl;

        return 0;
}
