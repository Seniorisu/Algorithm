# Markdown
### Markdown教程
[Markdown教程](https://markdown.com.cn/basic-syntax/)
***
***
# 数据结构
### ST表
- 常数优化->见树上倍增
- ST表实现(以区间最大值为例)
```c++
for(int j=1;j<=lg[n];j++){
	for(int i=1;(i+(1<<j)-1)<=n;i++)
		st[i][j]=max(st[i][j-1],st[i+(1<<(j-1))][j-1]);
}
```
- RMQ求解
```c++
int rmq(int x,int y){
	int k=lg[y-x+1];
	return max(st[x][k],st[y-(1<<k)+1][k]);
}
```
### 数组模拟邻接表
[数组模拟邻接表](https://www.acwing.com/blog/content/4663/)
链表的邻接表和图的邻接表均使用头插法。
- 边结点
```c++
struct Edge{
	int end;//边的终点
	int next;//下一个边节点
};
```
- 模拟邻接表数组
```c++
int head[500005]={};//头结点
Edge e[500005<<1]={};//边结点
int esum=1;//下一个边结点的序号
```
- 头插法插入边
```c++
void add(int x,int y){
	e[esum].end=y;
	e[esum].next=head[x];
	head[x]=esum++;
}
```
### 并查集
[并查集](https://blog.csdn.net/the_zed/article/details/105126583)
- find函数
```c++
//无优化
int find(int x){
	while(pre[x]!=x)x=pre[x];
	return x;
}
//路径压缩优化一
int find(int x){
	if(pre[x]==x)return x;
	return pre[x]=find(pre[x]);
}
```
- join函数
```c++
//无优化
void join(int x,int y){
	int fx=find(x),fy=find(y);
	if(fx!=fy)pre[fx]=fy;
}
//路径压缩优化二
void join(int x,int y){
	int fx=find(x),fy=find(y);
	if(fx!=fy){
		if(depth[fx]>depth[fy])pre[fy]=fx;
		else{
			if(depth[fx]==depth[fy])depth[fy]++;
			pre[fx]=fy;
		}
	}
}
```
[带权并查集]()

### 线段树
[线段树](https://blog.csdn.net/weixin_45697774/article/details/104274713)
[线段树-区间更新](https://blog.csdn.net/weq2011/article/details/128791426)
- 线段树节点
```c++
struct Tree{
	int l,r;
	int sum;
};
```
- build函数建树
```c++
void build(int id,int l,int r){
	tree[id].l=l,tree[id].r=r;
	if(l==r){
		tree[id].sum=arr[l];
		return;
	}
	int mid=(l+r)>>1;
	build(id*2,l,mid);
	build(id*2+1,mid+1,r);
	tree[id].sum=tree[id*2].sum+tree[id*2+1].sum;
}
```
- 区间查询find
```c++
int find(int id,int l,int r){
	if(tree[id].l>=l && tree[id].r<=r)return tree[id].sum;
	else if(tree[id].l>r || tree[id].r<l)return 0;
	else return find(id*2,l,r)+find(id*2+1,l,r);
}
```
- 单点修改add
```c++
void add(int id,int dis,int change){
	if(tree[id].l==tree[id].r){
		tree[id].sum+=change;
		return; 
	}
	if(dis<=tree[id*2].r)add(id*2,dis,change);
	else add(id*2+1,dis,change);
	tree[id].sum=tree[id*2].sum+tree[id*2+1].sum;
}
```

### 树状数组
[树状数组](https://blog.csdn.net/AAMahone/article/details/87746708)

### 树的直径
[两次DFS](https://blog.csdn.net/kaka03200/article/details/106576387)
[树形DP](https://blog.csdn.net/hnjzsyjyj/article/details/140254954)
- 树形DP
```c++
void dfs(int now,int fath){
    dp[now][0]=dp[now][1]=0;

    for(int i=head[now];i;i=e[i].nex){
        if(e[i].end!=fath){
            dfs(e[i].end,now);
            if((dp[e[i].end][0]+e[i].wei)>dp[now][0]){
                dp[now][1]=dp[now][0];
                dp[now][0]=dp[e[i].end][0]+e[i].wei;
            }
            else if((dp[e[i].end][0]+e[i].wei)>dp[now][1])
                dp[now][1]=dp[e[i].end][0]+e[i].wei;
        }
    }
    ans=max(ans,dp[now][0]+dp[now][1]);
}
```
***
***
# 算法

### 平年、闰年
- 每月天数
```c++
int ping_year[12]={31,28,31,30,31,30,31,31,30,31,30,31};
int run_year[12]={31,29,31,30,31,30,31,31,30,31,30,31};
```
- 判断方式
```c++
if((year%400==0)||(year%4==0&&year%100!=0))
```
***
### 前缀和、差分
[前缀和与差分](https://blog.csdn.net/weixin_45629285/article/details/111146240)
- 二维差分函数insert
```c++
void insert(int i,int j,int x,int y,int c){
	diff[i][j]+=c;
	diff[i][y+1]-=c;
	diff[x+1][j]-=c;
	diff[x+1][y+1]+=c;
}
```
[树上前缀和]()
[树上差分]()
***
## 快速幂
### 快速幂
[快速幂](https://blog.csdn.net/qq_19782019/article/details/85621386)
- 快速幂函数fast_power
```c++
long long fast_power(long long base,long long power){
	long long ans=1;
	while(power){
		if(power & 1)ans*=base;
		base*=base;
		power>>=1;
	}
	return ans;
}
```
### 大数取模
- 取模运算法则
> 1.(a + b) % p = ( a % p + b % p ) % p
> 2.(a - b) % p = ( a % p - b % p ) % p
> 3.(a * b) % p = ( a % p * b % p ) % p

### 矩阵快速幂
[矩阵快速幂](https://blog.csdn.net/gwk1234567/article/details/106444071)
斐波那契数列
***

## 双指针算法
### 滑动窗口

## 最近公共祖先(LCA)
[LCA](https://www.luogu.com.cn/problem/solution/P3379)
### 树上倍增
使用数组模拟邻接表

- 需要数据
```c++
int depth[500005]={-1},fa[500005][22]={},lg[500005]={};
//depth为每个节点的深度，根节点深度为0，根节点是depth[1]
//fa[i][j]为i号节点向上的第2^j个父节点
```
- 常数优化
```c++
for(int i=1;i<=n;i++)
	lg[i]=lg[i-1]+((1<<lg[i-1])==i);
//lg[i]=log2i+1
for(int i=1;i<=n;i++)
	lg[i]=lg[i-1]+((2<<lg[i-1])==i);
//lg[i]=log2i
```
- dfs预处理depth和fa数组
```c++
void dfs(int now,int fath){
	fa[now][0]=fath,depth[now]=depth[fath]+1;

	for(int j=1;j<lg[depth[now]];j++)fa[now][j]=fa[fa[now][j-1]][j-1];

	for(int i=head[now];i;i=e[i].next){
		if(e[i].end!=fath)dfs(e[i].end,now);
	}
}
```
- LCA
```c++
int LCA(int x,int y){
	if(depth[x]<depth[y])swap(x,y);//假设depth[x]>=depth[y]

	while(depth[x]>depth[y])
		x=fa[x][lg[depth[x]-depth[y]]-1];

	if(x==y)return y;

	for(int i=lg[depth[x]]-1;i>=0;i--){
		if(fa[x][i]!=fa[y][i])
			x=fa[x][i],y=fa[y][i];
	}
	return fa[x][0];
}
```

## 动态规划
### 树形DP
[树形DP、换根DP](https://blog.csdn.net/write_1m_lines/article/details/126263935)
1. 使用数组模拟邻接表
2. 先计算叶子节点，再计算根节点
- dfs
```c++
void dfs(int now,int fath){
    dp[now][0]=0;
    dp[now][1]=head[now].wei; //初始化

    for(int i=head[now].nex;i;i=edge[i].nex){
        if(edge[i].end!=fath){ //遍历子节点
            dfs(edge[i].end,now); //先计算子节点
            dp[now][0]=max(dp[now][0],max(dp[edge[i].end][0],dp[edge[i].end][1]));
            dp[now][1]=dp[now][1]+max((long long)0,dp[edge[i].end][1]); //更新父节点
        }
    }
}
```

## 字符串
### KMP算法
[KMP模式匹配算法](https://blog.csdn.net/m0_58386652/article/details/144316661)

## 图论
### 最小生成树
#### prime算法
