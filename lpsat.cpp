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

typedef long double scalar;
typedef Matrix<scalar, Dynamic, Dynamic> mat;
const scalar one = 1, two = 2, half = .5, three = 3, four = 4;

#define HALLEY
#define CLASSIC

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

	scalar	va = (a > 0 ? x(a-1,0) : x(-a-1,0)),
		vb = (b > 0 ? x(b-1,0) : x(-b-1,0)),
		vc = (c > 0 ? x(c-1,0) : x(-c-1,0));

#ifdef CLASSIC
	scalar 	_a = (a > 0 ? one - va : va),
		_b = (b > 0 ? one - vb : vb),
		_c = (c > 0 ? one - vc : vc);

	scalar	ga = -sgn(a),
		gb = -sgn(b),
		gc = -sgn(c);

	scalar	gga = 0,
		ggb = 0,
		ggc = 0;
#else
	scalar 	da = one / (one + va * va),
		db = one / (one + vb * vb),
		dc = one / (one + vc * vc);

	scalar 	_a = (a > 0 ? one - va : va) * da,
		_b = (b > 0 ? one - vb : vb) * db,
		_c = (c > 0 ? one - vc : vc) * dc;

	scalar	ga = (a > 0 ? va * va - va * two - one : (one - va) * (one + va) ) * da * da,
		gb = (b > 0 ? vb * vb - vb * two - one : (one - vb) * (one + vb) ) * db * db,
		gc = (c > 0 ? vc * vc - vc * two - one : (one - vc) * (one + vc) ) * dc * dc;

	scalar	gga = (a > 0 ? two * (one + va) * (va * va - four * va + one) : two * va * (va * va - three) ) * da * da * da,
		ggb = (b > 0 ? two * (one + vb) * (vb * vb - four * vb + one) : two * vb * (vb * vb - three) ) * db * db * db,
		ggc = (c > 0 ? two * (one + vc) * (vc * vc - four * vc + one) : two * vc * (vc * vc - three) ) * dc * dc * dc;
#endif
	uint aa = abs(a) - 1, ab = abs(b) - 1, ac = abs(c) - 1;

	g(0, aa) = ga * _b * _c;
	g(0, ab) = gb * _a * _c;
	g(0, ac) = gc * _a * _b;
#ifdef HALLEY
	H = mat::Zero(x.rows(), x.rows());
	H(aa, aa) = gga;
	H(ab, ab) = ggb;
	H(ac, ac) = ggc;
	H(aa, ab) = H(ab, aa) = ga * gb * _c;
	H(aa, ac) = H(ac, aa) = ga * gc * _b;
	H(ab, ac) = H(ac, ab) = gb * gc * _a;
#endif
	return _a * _b * _c;
}

bool eval(const mat& m, const mat& x, bool print = false) {
	mat g, H;
	scalar r = 1;
	for (uint n = 0; n < m.rows(); n++) {
		if ( eval(m(n, 0), m(n, 1), m(n, 2), round(x), g, H) ) { if (print) cout<<0<<' '; return false; }
		else if (print) cout<<1<<' '; 
	}
	return true;
}

void read(istream& is, uint iters, uint print, uint batches, const char* fname = 0) {
	string str;
        uint rows, cols, n = 0, batch = 0;
	scalar minj = HUGE_VAL, minf = HUGE_VAL, mins = HUGE_VAL, sn, minhg = -HUGE_VAL, jn, fn, hgn, hn, minhn = -HUGE_VAL;
	int v;
	do { getline(is, str); } while (str[0] == 'c');
        sscanf(str.c_str(), "p cnf %d %d", &cols, &rows);

	mat	//m = mat::Zero(rows, cols),
		&J = *new mat(mat::Zero(rows + cols, cols)), g,
		&F = *new mat(mat::Zero(rows + cols, 1)),
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

	mat step, sv, sv1;
	mat *H = new mat[F.rows()];
	scalar x0;
	do {
		x = mat::Ones(cols, 1);
		switch (batch++) {
			case 0: x0 = .5; break;
			case 1: x0 = 1; break;
			case 2: x0 = 0; break;
//			case 3: x *= 1./3.; break;
//			case 4: x *= 2./3.; break;
//			case 5: x *= .25; break;
//			case 6: x *= .75; break;
			default: x0 = scalar(batch)/scalar(batches); break;
		}
		x *= x0;
		for (uint i = 1; i <= iters; i++) {
		        for (n = 0; n < rows; n++) {
				F(n, 0) = eval(/*m*/D(n,0),D(n,1),D(n,2), x, g, H[n]);
				J.row(n) = g;
			}
		        for (n = rows; n < rows + cols; n++) {
				scalar t = x(n - rows, 0);
				F(n, 0) = /*pow(*/t * (one - t) * half * pow(half, cols);
				J(n, n - rows) = (one - two * t) * half * pow(half, cols);
#ifdef HALLEY
				H[n] = mat::Zero(x.rows(), x.rows());
				H[n](n - rows, n - rows) = -one * pow(half, cols);
#endif
			}
// https://www8.cs.umu.se/~viklands/tensor.pdf
			JacobiSVD<mat> svd(J, ComputeFullU | ComputeFullV);
			step = -svd.solve(F);
//#ifdef HALLEY
			mat Hg(F.rows(), step.rows());
			for (uint j = 0; j < Hg.rows(); j++) Hg.row(j) = step.transpose() * H[j];
			JacobiSVD<mat> svd2(J + Hg, ComputeFullU | ComputeFullV);
			x += step - svd2.solve(Hg * step) / two;
			for (uint j = 0; j < x.rows(); j++) if (x(j, 0) < -1) x(j, 0) = -1; else if (x(j,0) > 2) x(j, 0) = 2; 
			bool found = eval(D, x);
			if (i == 1/* || found*/) {
				scalar 	normhk = step.norm(), 
					M = sqrt((Hg.transpose() * Hg).trace());
				mat dd = mat::Zero(J.cols(), J.rows());
				for (uint rr = 0; rr < min(dd.rows(), dd.cols()); rr++)
					dd(rr, rr) = svd.singularValues()(rr) ? one / svd.singularValues()(rr) : 0;
				mat jinv = svd.matrixV() * dd * svd.matrixU().transpose();
//				cout<< <<endl;
				scalar alpha = M * sqrt((jinv.transpose() * jinv).trace()) * normhk;
				if (i%print == 0) cout<<fname<<" alpha: "<< M * sqrt((jinv.transpose() * jinv).trace()) * normhk<< "\tx0: "<<x0<<endl;
			//	if (alpha > D.rows()) break;
			}
//#endif
			for (uint j = hgn = 0; j < F.rows(); hgn += H[j++].squaredNorm()) ;
			minj = min(minj, jn = J.norm());
			minf = min(minf, fn = F.norm());
			mins = min(mins, sn = step.norm());
			minhg = max(minhg, hgn = sqrt(hgn));
			minhn = max(minhn, hn = Hg.norm());

			if (found || (i%print == 0)) { 
				if (fname) { if (found) cout<<"found solution\t"<<fname<<endl; }
				cout	<< "F: " << F.transpose() <<endl << endl
					<< "x: " << x.transpose() <<endl << endl
					<< "step: " << step.transpose() << endl << endl;
				cout << "output: "; eval(D,x,true);
				cout	<< "batch " << batch
					<<" iteration " <<i
					<<"\t||J||: "<<jn
					<<"\t||F||: "<<fn
					<<"\t||step||: " << sn 
					<< "\t||Hg||: " << hgn 
					<< "\t||H||: " << hn<<endl;
//					<< mat(svd.singularValues()).transpose()<<endl
//					<< mat(svd2.singularValues()).transpose()<<endl;
				if (found) exit(0); 
			}
			if (sn < 1e-7) {
				sv = mat(svd.singularValues()).transpose();
				sv1 = mat(svd2.singularValues()).transpose();
				break;
			}
		}
//        	if (fname) cout<<fname<<'\t';
//	        cout <<"batch: "<<batch<< "\tsatness: " << d /*d * 2*/ <<endl;
	} while (batch < batches); 
	if (fname) cout<<"solution not found\t"<<fname<<endl;
/*	if (fname) cout<<fname<<'\t';
	cout 	<< "min||J||: " << minj
		<<"\tmin||F||: " << minf 
		<< "\tmin||step||: " << mins 
		<< "\tx error: " << sqrt(((-x.transpose()*x+x.transpose()*mat::Ones(x.rows(), x.cols())).norm())/scalar(x.rows()))
		<< "\tmax||Hg||: "<< minhg 
		<<"\tmax||H||:"<<minhn <<endl<<sv<<endl<<sv1<<endl;
*/}

int main(int argc, char** argv) {
	if (argc < 4) { 
		cout	<< "Usage: <iterations> <print every> <batches> [ <file1> <file2> ... ] " << endl
			<< "stdin is used if no files specified."<<endl;
		return 1; 
	}
	int ws;
	pid_t pid;
	vector<pid_t> waitlist;
	std::cout << std::setprecision(2);
	if (argc == 3) read(cin, atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));
	else {
		for (uint n = 4; n < argc; n++) {
			if (!(pid = fork())) {
				read(*new ifstream(argv[n]), atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), argv[n]);
				return 0;
			}
			waitlist.push_back(pid);
			if ((n-3)%7 == 0) {
				//for (int p : waitlist) 
				waitpid(*waitlist.begin(), &ws, 0);
				waitlist.erase(waitlist.begin());
//				waitlist.clear();
			}
		}
		for (int p : waitlist) waitpid(p, &ws, 0);
	}
	return 0;
}
