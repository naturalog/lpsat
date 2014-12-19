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
#include <Eigen/Sparse>

using namespace std;
using namespace Eigen;

typedef float scalar;
typedef Matrix<scalar, Dynamic, Dynamic> mat;
const scalar one = 1, two = 2;

#define HALLEY

inline mat round(const mat& x) {
	mat r = x;
	for (uint n = 0; n < x.rows(); n++)     
		for (uint k = 0; k < x.cols(); k++)
			r(n,k) = (r(n,k) > one/two ? 1 : 0);
	return r;
}

// note: assuming variable cannot appear more than once at the same clause
/*inline scalar eval(const mat& clause, const mat& x, mat& g) {
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
}*/
template<typename T> T sgn(const T& t) { return t>0?1:-1; }
inline scalar eval(int a, int b, int c, const mat& x, mat& g, mat& H) {
        g = mat::Zero(1, x.rows());

	scalar 	_a = (a > 0 ? one - x(a-1,0) : x(-a-1,0)),
		_b = (b > 0 ? one - x(b-1,0) : x(-b-1,0)),
		_c = (c > 0 ? one - x(c-1,0) : x(-c-1,0));

	g(0, abs(a) - 1) = -sgn(a) * _b * _c;
	g(0, abs(b) - 1) = -sgn(b) * _a * _c;
	g(0, abs(c) - 1) = -sgn(c) * _a * _b;
#ifdef HALLEY
	H = mat::Zero(x.rows(), x.rows());
	H(abs(a) - 1, abs(a) - 1) = H(abs(b) - 1, abs(b) - 1) = H(abs(c) - 1, abs(c) - 1) = 0;
	H(abs(a) - 1, abs(b) - 1) = H(abs(b) - 1, abs(a) - 1) = sgn(a)*sgn(b)*_c;
	H(abs(a) - 1, abs(c) - 1) = H(abs(c) - 1, abs(a) - 1) = sgn(a)*sgn(c)*_b;
	H(abs(b) - 1, abs(c) - 1) = H(abs(c) - 1, abs(b) - 1) = sgn(b)*sgn(c)*_a;
#endif
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
	scalar minj = HUGE_VAL, minf = HUGE_VAL, mins = HUGE_VAL, sn, minhg = HUGE_VAL, jn, fn, hgn, hn, minhn = HUGE_VAL;
	int v;
	do { getline(is, str); } while (str[0] == 'c');
        sscanf(str.c_str(), "p cnf %d %d", &cols, &rows);

	mat	//m = mat::Zero(rows, cols),
		&J = *new mat(mat::Zero(rows + cols, cols)), g;
	mat	&F = *new mat(mat::Zero(rows + cols, 1)),
		&r = *new mat(mat::Ones(rows, 1)),
		&D = *new mat(mat::Zero(rows, 3)),
		x;

        for (; n < rows; n++) {
                getline(is, str);
		uint j = 0;
		for (stringstream ss(str); !ss.eof();) {
			ss >> v;
			if (v) /*m(n, abs(v) - 1) =*/ ((D(n, j++)=v) > 0 ? 1 : -1);
		}
        }

	mat step;
	mat *H = new mat[F.rows()];
	do {
		x = mat::Ones(cols, 1);// * fabs(batch % 2 ? one - pow(3./4.,(batch-1)/2) : pow(3./4.,batch/2));
		switch (batch++) {
			case 0: x *= 0; break;
			case 1: x *= 1; break;
			case 2: x *= .5; break;
			case 3: x *= 1./3.; break;
			case 4: x *= 2./3.; break;
			case 5: x *= .25; break;
			case 6: x *= .75; break;
			default: return;
		}
		for (uint i = 1; i <= iters; i++) {
		        for (n = 0; n < rows; n++) {
				F(n, 0) = eval(/*m*/D(n,0),D(n,1),D(n,2), x, g, H[n]);
				J.row(n) = g;
			}
		        for (n = rows; n < rows + cols; n++) {
				scalar t = x(n - rows, 0);
					F(n, 0) = /*pow(*/t * (one - t);
					J(n, n - rows) = (one - two * t);
#ifdef HALLEY
					H[n] = mat::Zero(x.rows(), x.rows());
					H[n](n - rows, n - rows) = -two;
#endif
			}
// https://www8.cs.umu.se/~viklands/tensor.pdf
			JacobiSVD<mat> svd(J, ComputeFullU | ComputeFullV);
			step = -svd.solve(F);
#ifdef HALLEY
			mat Hg(F.rows(), step.rows());
			for (uint j = 0; j < Hg.rows(); j++) 
//				for (uint l = 0; l < Hg.cols(); l++) 
//					Hg(j, l) = (step.transpose() * H[j].col(l))(0,0);
					Hg.row(j) = step.transpose() * H[j];
			JacobiSVD<mat> svd2(J + Hg, ComputeFullU | ComputeFullV);
			step -= svd2.solve(Hg * step) / two;
#endif
			x += step;
			if (i % print == 0) 
				cout<<endl<<F.transpose()<<endl
					<<endl<<x.transpose()<<endl;
			hgn = 0;
			for (uint j = 0; j < F.rows(); j++) hgn += H[j].squaredNorm();
			minj = min(minj, jn = J.norm());
			minf = min(minf, fn = F.norm());
			mins = min(mins, sn = step.norm());
			minhg = min(minhg, hgn = sqrt(hgn));
			minhn = min(minhn, hn = Hg.norm());

			if (eval(D, x)) { 
				if (fname) cout<<fname<<'\t'; 
				cout<<"solution found batch "<<batch<<" iteration "<<i<<"\t||J||: "<<jn<<"\t||F||: "<<fn<<"\t||step||: " << sn << "\t||Hg||: " << hgn << "\t||H: " << hn<<endl; 
				return; 
			}
			if (sn < 1e-4) break;
		}
//        	if (fname) cout<<fname<<'\t';
//	        cout <<"batch: "<<batch<< "\tsatness: " << d /*d * 2*/ <<endl;
	} while (batch < 6); 
	if (fname) cout<<fname<<'\t';
	cout << "min||J||: " << minj<<"\tmin||F||: " <<minf << "\tmin||step||: " << mins << "\tx error: " << sqrt(((-x.transpose()*x+x.transpose()*mat::Ones(x.rows(), x.cols())).norm())/scalar(x.rows()))<< "\tmin||Hg||: "<<minhg <<"\tmin||H||:"<<minhn <<endl;
}

int main(int argc, char** argv) {
	if (argc < 3) return 1;
	int ws;
	pid_t pid;
	vector<pid_t> waitlist;
	std::cout << std::setprecision(2);
	if (argc == 3) read(cin, atoi(argv[1]), atoi(argv[2]));
	else {
		for (uint n = 3; n < argc; n++) {
	//		if (!(pid = fork())) {
				read(*new ifstream(argv[n]), atoi(argv[1]), atoi(argv[2]), argv[n]);
	//			return 0;
	//		}
	//		waitlist.push_back(pid);
//			if ((n-2)%7 == 0) {
//				for (int p : waitlist) waitpid(p, &ws, 0);
//				waitlist.clear();
//			}
		}
		for (int p : waitlist) waitpid(p, &ws, 0);
	}
	return 0;
}
