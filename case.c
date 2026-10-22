// I/O
// 6821f502f5b6ec95c505e5dd8f3cd5d3
#include<stdio.h>
#define For(i) for(i=1;i<=9;i++)
int main()
{
    int x,y,z,t,c1,c2,r1,r2,d1,d2;
    scanf("%d%d%d%d%d%d",&r1,&r2,&c1,&c2,&d1,&d2);
    For(x)For(y)For(z)For(t)if(x+y==c1&&z+t==c2&&x+z==r1&&y+t==r2&&x+t==d1&&y+z==d2&&x!=y&&x!=z&&x!=t&&y!=z&&y!=t&&z!=t){printf("%d %d\n%d %d",x,z,y,t);return 0;}puts("-1");
    return 0;
}

// array
// 3e1751a2990134f2132d743afe02a10e
#include <stdio.h>

int prime[1005000];
int primeList[1005000];

void init() {
    int i, cnt = 0, j;
    prime[2] = 1;
    primeList[cnt++] = 2;
    for(i = 3; i < 1005000; i += 2) {
        for(j = 0; j < cnt; j++) {
            if(i % primeList[j] == 0) break;
            if((1ll * primeList[j]) * primeList[j] > i) break;
        }
        if(j == cnt || i % primeList[j] != 0) {
            primeList[cnt++] = i;
            prime[i] = 1;
        }
    }
}

// Mixtype
// 52c6aa73ff4460799402c646c6263630
/* practice with Dukkha */
#include <stdio.h>

#define N	100
#define X	100
#define MD	1000000007

long long power(int a, int k) {
	long long p;

	if (k == 0)
		return 1;
	p = power(a, k >> 1);
	p = p * p % MD;
	if (k & 1)
		p = p * a % MD;
	return p;
}

int main() {
	static int ff[X + 1], ds[X + 1][N][N + 1], dp[X + 1][N], dq[X + 1][N], dr[X + 1];
	static char cc[N + 1];
	int n, x, i, l, r, m;

	scanf("%d%d%s", &n, &x, cc);
	if (x == 0) {
		printf("%d\n", n == 1 && cc[0] == '0');
		return 0;
	}
	ff[0] = ff[1] = 1;
	for (i = 2; i <= x; i++)
		ff[i] = (ff[i - 1] + ff[i - 2]) % (MD - 1);
	for (i = 0; i <= x; i++)
		ff[i] = power(2, ff[i]);
	for (l = 0; l < n; l++) {
		ds[0][l][l] = ds[1][l][l] = 1;
		ds[0][l][l + 1] = cc[l] == '0' ? 1 : 0;
		ds[1][l][l + 1] = cc[l] == '1' ? 1 : 0;
	}
	for (i = 2; i <= x; i++)
		for (l = 0; l < n; l++)
			for (r = l; r < n; r++) {
				long long z = 0;

				for (m = l; m <= r; m++)
					z = (z + (long long) ds[i - 1][l][m] * ds[i - 2][m][r]) % MD;
				ds[i][l][r] = z;
			}
	if (n > 1) {
		dp[0][n - 1] = ds[0][n - 1][n];
		dp[1][n - 1] = ds[1][n - 1][n];
		for (i = 2; i <= x; i++)
			for (l = 1; l < n; l++) {
				long long z = (long long) dp[i - 1][l] * ff[i - 2] % MD;

				for (m = l; m < n; m++)
					z = (z + (long long) ds[i - 1][l][m] * dp[i - 2][m]) % MD;
				dp[i][l] = z;
			}
		dq[0][1] = ds[0][0][1];
		dq[1][1] = ds[1][0][1];
		for (i = 2; i <= x; i++)
			for (r = 1; r < n; r++) {
				long long z = (long long) ff[i - 1] * dq[i - 2][r] % MD;

				for (m = r; m > 0; m--)
					z = (z + (long long) dq[i - 1][m] * ds[i - 2][m][r]) % MD;
				dq[i][r] = z;
			}
	}
	dr[0] = ds[0][0][n];
	dr[1] = ds[1][0][n];
	for (i = 2; i <= x; i++) {
		long long z = ((long long) dr[i - 1] * ff[i - 2] + (long long) ff[i - 1] * dr[i - 2]) % MD;

		for (m = 1; m < n; m++)
			z = (z + (long long) dq[i - 1][m] * dp[i - 2][m]) % MD;
		dr[i] = z;
	}
	printf("%d\n", dr[x]);
	return 0;
}

// pointers
#include<stdio.h>
#include<stdlib.h>

int main()
{
	int i,temp=0,n,k,x,*arr;
	scanf("%d%d%d",&n,&k,&x);
	arr=malloc(n*sizeof(int));
	for(i=0;i<n;i++)
	{
		scanf("%d",&arr[i]);
	}
	for(i=n-k;i<n;i++)
	{
		arr[i]=x;
	}
	for(i=0;i<n;i++)
		temp+=arr[i];
	printf("%d\n",temp);
}

// summary
// cd0e90042a6aca647465f1d51e6dffc4
#include<stdio.h>
int main() {
    int n,a,b,i,m,sum;
    scanf("%d %d %d",&n,&a,&b);
    if(b>=0)
        {
            m=a;
        for(i=1; i<=b; i++) {
            m++;
            if(m>n)
                m=1;
        }
        printf("%d",m);
        }
        else
        {
            m=a;
            for(i=1;i<=abs(b);i++)
            {
                m--;
                if(m==0)
                    m=n;
            }printf("%d",m);
        }
}

