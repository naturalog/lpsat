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

inline mat round(const mat& x) {
	mat r = x;
	for (uint n = 0; n < x.rows(); n++)     
		for (uint k = 0; k < x.cols(); k++)
			r(n,k) = (r(n,k) > one/two ? 1 : 0);
	return r;
}

// note: assuming variable cannot appear more than once at the same clause
inline scalar eval(const mat& clause, const mat& x, mat& g) {
	scalar r = one, p, xn;
	g = mat::Ones(1, x.rows());
	for (uint n = 0; n < clause.cols(); n++) 
		if (clause(0, n)) {
			xn = x(n, 0);
			r *= (p = (clause(0, n) > 0 ? one - xn : xn));
			g(0, n) *= -clause(0, n);
			for (uint k = 0; k < clause.cols(); k++)
				if (n != k) g(0, k) *= p;
		} else g(0, n) = 0;
//	g *= r;
	return r;// * r / two;
}
template<typename T> T sgn(const T& t) { return t>0?1:-1; }
inline scalar eval(int a, int b, int c, const mat& x, mat& g, mat& H) {
        g = mat::Zero(1, x.rows());

	scalar 	_a = (a > 0 ? one - x(a-1,0) : x(-a-1,0)),
		_b = (b > 0 ? one - x(b-1,0) : x(-b-1,0)),
		_c = (c > 0 ? one - x(c-1,0) : x(-c-1,0));

	g(0, abs(a) - 1) = (a > 0 ? -_b * _c : _b * _c);
	g(0, abs(b) - 1) = (b > 0 ? -_a * _c : _a * _c);
	g(0, abs(c) - 1) = (c > 0 ? -_a * _b : _a * _b);

	H = mat::Zero(x.rows(), x.rows());
	H(abs(a) - 1, abs(a) - 1) = H(abs(b) - 1, abs(b) - 1) = H(abs(c) - 1, abs(c) - 1) = 0;
	H(abs(a) - 1, abs(b) - 1) = H(abs(b) - 1, abs(a) - 1) = sgn(a)*sgn(b)*_c;
	H(abs(a) - 1, abs(c) - 1) = H(abs(c) - 1, abs(a) - 1) = sgn(a)*sgn(c)*_b;
	H(abs(b) - 1, abs(c) - 1) = H(abs(c) - 1, abs(b) - 1) = sgn(b)*sgn(c)*_a;

	return _a * _b * _c;
}


bool eval(const mat& m, const mat& x) {
	mat g, H;
	scalar r = 1;
	for (uint n = 0; n < m.rows(); n++) r *= one - eval(m(n,0),m(n,1),m(n,2),round(x),g, H);
	return r == 1;
}

void read(istream& is, uint iters, uint print, const char* fname = 0) {
	string str;
        uint rows, cols, n = 0, batch = 0;
	scalar d = HUGE_VAL;
	int v;
	do { getline(is, str); } while (str[0] == 'c');
        sscanf(str.c_str(), "p cnf %d %d", &cols, &rows);

	mat	//m = mat::Zero(rows, cols),
		J = mat::Zero(rows + cols, cols),
		F = mat::Zero(rows + cols, 1),
		r = mat::Ones(rows, 1),
		D = mat::Zero(rows, 3),
		g, x, H;

        for (; n < rows; n++) {
                getline(is, str);
		uint j = 0;
		for (stringstream ss(str); !ss.eof();) {
			ss >> v;
			if (v) /*m(n, abs(v) - 1) =*/ ((D(n, j++)=v) > 0 ? 1 : -1);
		}
        }

	do {
		batch++;
		x = mat::Ones(cols, 1) * fabs(batch % 2 ? one - pow(7./8.,(batch-1)/2) : pow(7./8.,batch/2));
		for (uint i = 1; i <= iters; i++) {
		        for (n = 0; n < rows; n++) {
				F(n, 0) = eval(/*m*/D(n,0),D(n,1),D(n,2), x, g, H);
				J.row(n) = g;
			}
		        for (n = rows; n < rows + cols; n++) {
				scalar t = x(n - rows, 0);
//			        for (uint k = 0; k < cols; k++) {
//					F(n, 0) = exp(t * (one - t)) - one;
//					J(n, n - rows) = (one - two * t) * (one + F(n, 0));
					F(n, 0) = /*pow(*/t * (one - t)/*,2)/two*/;
					J(n, n - rows) = (one - two * t);// * (t * (one - t));
//					F(n, 0) += pow(t,2*(n-rows+1))*pow(one-t,2*k+2)/scalar(x.rows());//t * (one - t);
//					J(n, n - rows) = (pow(t,2*(n-rows)+1)*pow(one-t,2*k+2)*scalar(2*(n-rows+1)) -
//							 pow(t,2*(n-rows+1))*pow(one-t,2*k+1)*scalar(2*k+2))/scalar(x.rows()) ;//one - two * t;
			//	}
			}
			JacobiSVD<mat> svd(J, ComputeFullU | ComputeFullV);
			x -= svd.solve(F);// / (two*two*two);
		//	x = round(x);
			if (i % print == 0) 
				cout<<endl<<F.transpose()<<endl
					<<endl<<x.transpose()<<endl;
			if (F.norm() < 1e-3) { if (fname) cout<<fname<<'\t'; cout<<"solution found\t"<<(eval(D, x) ? "verified" : "bad eps")<<endl; exit(0); }
			if (J.norm() < 1e-3) break;
		}
		d = min(d, F.norm());
//        	if (fname) cout<<fname<<'\t';
//	        cout <<"batch: "<<batch<< "\tsatness: " << d /*d * 2*/ <<endl;
	} while (batch < 50); 
	if (fname) cout<<fname<<'\t';
	cout << "final error: " << d /*d * 2*/ << "\t x error: " << /*(x - round(x)).norm()*/sqrt(((-x.transpose()*x+x.transpose()*mat::Ones(x.rows(), x.cols())).norm())/scalar(x.rows()))<<endl;
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
