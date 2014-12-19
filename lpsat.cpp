#include <cstring>
#include <iostream>
#include <cstdlib>
#include <string>
#include <sstream>
#include <Eigen/Dense>
#include <eigen3/Eigen/SVD>
#include <iomanip>
#include <fstream>

using namespace std;
using namespace Eigen;

typedef float scalar;
typedef Matrix<scalar, Dynamic, Dynamic> mat;
const scalar one = 1, two = 2;

// note: assuming variable cannot appear more than once at the same clause
scalar eval(const mat& clause, const mat& x, mat& g) {
	scalar r = one, p;
	g = mat::Ones(1, x.rows());
	for (uint n = 0; n < clause.cols(); n++) 
		if (clause(0, n)) {
			r *= (p = (clause(0, n) > 0 ? one - x(n, 0) : x(n, 0)));
			g(0, n) *= -clause(0, n);
			for (uint k = 0; k < clause.cols(); k++)
				if (n != k) g(0, k) *= p;
		} else g(0, n) = 0;
	return r;
}

void read(istream& is, uint iters, uint print) {
	string str;
        uint rows, cols, n = 0, batch = 0;
	scalar d = 1;
	int v;
	do { getline(is, str); } while (str[0] == 'c');
        sscanf(str.c_str(), "p cnf %d %d", &cols, &rows);

	mat	m = mat::Zero(rows, cols),
		J = mat::Zero(rows + cols, cols),
		F = mat::Zero(rows + cols, 1),
		r = mat::Ones(rows, 1),
		g;

        for (; n < rows; n++) {
                getline(is, str);
		for (stringstream ss(str); !ss.eof();) {
			ss >> v;
			if (v) m(n, abs(v) - 1) = (v > 0 ? 1 : -1);
		}
        }

	do {
		batch++;
		mat x = mat::Ones(cols, 1) * (batch % 2 ? one - pow(one/two,(batch-1)/2) : pow(one/two,batch/2));
		for (uint i = 1; i <= iters; i++) {
		        for (n = 0; n < rows; n++) {
				F(n, 0) = eval(m.row(n), x, g);
				J.row(n) = g;
			}
		        for (n = rows; n < rows + cols; n++) {
				scalar t = x(n - rows, 0);
				F(n, 0) = t * (one - t);
				J(n, n - rows) = one - two * t;
			}
			JacobiSVD<mat> svd(J, ComputeFullU | ComputeFullV);
			x -= svd.solve(F);
			if (i % print == 0) 
				cout<<endl<<F.transpose()<<endl
					<<endl<<x.transpose()<<endl;
			if (F.norm() < 1e-3) { cout<<"solution found"<<endl; return; }
		}
		d = min(d, one / F.norm());
	} while (batch < 10); 
	cout << "satness: " << d /*d * 2*/ <<endl;
}

int main(int argc, char** argv) {
	if (argc != 3 && argc != 4) return 1;
	if (argc == 4) cout<<argv[3]<<'\t';
	std::cout << std::setprecision(2);
	read(argc != 4 ? cin : *new ifstream(argv[3]), atoi(argv[1]), atoi(argv[2]));
	return 0;
}
