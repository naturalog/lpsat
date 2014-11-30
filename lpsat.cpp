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

pair<mat, mat> dimacs2eigen(istream& is) {
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
	cout<<a<<endl;
	cout<<m<<endl;
	cout<<a<<endl;
	return make_pair(m, rhs);
}

mat cnf;

mat objective(mat x, mat& j) {
	mat r(cnf*x);
	j = cnf.transpose();
	return r;
}

int main(int argc,char** argv){
	auto p = dimacs2eigen(cin);
	cnf = p.first;
	mat x = mat::Ones(cnf.cols(), 1);
	mat j = mat::Zero(cnf.cols(), cnf.rows());
	cout << "F:" << endl <<(cnf * x).transpose()<<endl<<endl
	     << "JT:" << endl << cnf <<endl<<endl
	     << "JTJ:" << endl << cnf.transpose() * cnf <<endl<<endl;
	JacobiSVD<mat> svd(cnf, ComputeThinU | ComputeThinV);
	cout << "D:" << endl << svd.singularValues() << endl;
//	cout<<j.squaredNorm()<<endl;
	cout << endl << svd.solve(p.second) << endl;

        return 0;
}
