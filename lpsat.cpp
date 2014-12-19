#include <cstring>
#include <iostream>
#include <cstdlib>
#include <string>
#include <sstream>
#include <Eigen/Dense>
#include <eigen3/Eigen/SVD>
#include <iomanip>
#include <fstream>
#include <sys/wait.h>

using namespace std;
using namespace Eigen;

typedef float scalar;
typedef Matrix<scalar, Dynamic, Dynamic> mat;
const scalar one = 1, two = 2;

// note: assuming variable cannot appear more than once at the same clause
inline scalar eval(const mat& clause, const mat& x, mat& g, bool round = false) {
	scalar r = one, p, xn;
	g = mat::Ones(1, x.rows());
	for (uint n = 0; n < clause.cols(); n++) 
		if (clause(0, n)) {
			xn = round ? (x(n, 0) < one/two ? 0 : 1): x(n, 0);
			r *= (p = (clause(0, n) > 0 ? one - xn : xn));
			g(0, n) *= -clause(0, n);
			for (uint k = 0; k < clause.cols(); k++)
				if (n != k) g(0, k) *= p;
		} else g(0, n) = 0;
	return round ? one - r : r;
}

bool eval(const mat& m, const mat& x) {
	mat g;
	scalar r = 1;
	for (uint n = 0; n < m.rows(); n++) r *= eval(m.row(n),x,g,true);
	return r == 1;
}

void read(istream& is, uint iters, uint print, const char* fname = 0) {
	string str;
        uint rows, cols, n = 0, batch = 0;
	scalar d = HUGE_VAL;
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
		mat x = mat::Ones(cols, 1) * fabs(batch % 2 ? one - pow(7./8.,(batch-1)/2) : pow(7./8.,batch/2));
		for (uint i = 1; i <= iters; i++) {
		        for (n = 0; n < rows; n++) {
				F(n, 0) = eval(m.row(n), x, g);
				J.row(n) = g;
			}
		        for (n = rows; n < rows + cols; n++) {
				scalar t = x(n - rows, 0);
//			        for (uint k = 0; k < cols; k++) {
					F(n, 0) = t * (one - t);
					J(n, n - rows) = one - two * t;
//					F(n, 0) += pow(t,2*(n-rows+1))*pow(one-t,2*k+2)/scalar(x.rows());//t * (one - t);
//					J(n, n - rows) = (pow(t,2*(n-rows)+1)*pow(one-t,2*k+2)*scalar(2*(n-rows+1)) -
//							 pow(t,2*(n-rows+1))*pow(one-t,2*k+1)*scalar(2*k+2))/scalar(x.rows()) ;//one - two * t;
			//	}
			}
			JacobiSVD<mat> svd(J, ComputeFullU | ComputeFullV);
			x -= svd.solve(F) / two;
			if (i % print == 0) 
				cout<<endl<<F.transpose()<<endl
					<<endl<<x.transpose()<<endl;
			if (F.norm() < 1e-3) { if (fname) cout<<fname<<'\t'; cout<<"solution found\t"<<(eval(m, x) ? "verified" : "bad eps")<<endl; exit(0); }
			if (J.norm() < 1e-3) break;
		}
		d = min(d, one / F.norm());
	} while (batch < 50); 
	if (fname) cout<<fname<<'\t';
	cout << "satness: " << d /*d * 2*/ <<endl;
	exit(0);
}

int main(int argc, char** argv) {
	if (argc < 3) return 1;
	int ws;
	pid_t pid;
	vector<pid_t> waitlist;
	std::cout << std::setprecision(2);
	if (argc == 3) read(cin, atoi(argv[1]), atoi(argv[2]));
	else {
		for (uint n = 3; n < argc; n++) 
			if (!(pid = fork())) 
				read(*new ifstream(argv[n]), atoi(argv[1]), atoi(argv[2]), argv[n]);
			else waitlist.push_back(pid);
		for (int p : waitlist) waitpid(p, &ws, 0);
	}
	return 0;
}
