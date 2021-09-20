

#define max 6
#define max2 1956

int fact (int n){
    if(n<1)
        return 1;

    else
        return n*fact(n-1);
}

int get_limit(int n){

    //return 325; // testing for 5 cities
    int sum=0;
    int k;

    for(k=1;k<=n;k++){
        sum = sum + fact(n)/fact(n-k);
    }
    printf("%d\n",sum);
    return sum;
}

int hash_lookup(int a[max],int len){

    int i=0;
    int hash_val=0;

    for(i=0;i<len;i++){
        hash_val=hash_val+i*pow(a[i],2);
    }
    return hash_val;
}

void  modify(int mod[],int len) //  traveled node is removed
{
    int j;
    for(j=0;j<len-1;j++)
    {
        mod[j]=mod[j+1];
    }
    mod[j]=0;
}

float min(float arr[], int l,int choice) // if choice =1--> returns index, else returns minimum distance
{
    float m=arr[0];
    int d,index;

    for(d=1;d<l;d++)
    {

        if(arr[d]<m)
        {
            m=arr[d];
            index=d;
        }
    }
    if(choice)
    return index ;
    else return m;
}

void copy( int a[], int b[], int l)
{
    int z;
    for(z=0;z<l;z++)
    {
        a[z]=b[z];
    }
}

void clear(int v[], int g)
{
    int z;
     for(z=0;z<g;z++)
     {
         v[z]=0;
     }
}

void append(int n,int array[],int lim)
{
    int y;
    for(y=1;y<lim;y++)
    {
        if(array[y]==0)
        {
            array[y]=n;
            break;
        }
    }
}
