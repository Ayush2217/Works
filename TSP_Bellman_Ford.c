#include<stdio.h>
#include<math.h>

#include "functions.c"
#include "display.c"

#define max 6
#define max2 1956


float bellman(int node, int prev_nodes[], int nc, float c[max][max], int rem_nodes[], int k , int o[max][max2]) // k-- length of the remaining nodes set
{

    int i=0, ind=0,m,sp;
    float r;
    float temp[k];
    int var[k];
    int next;
    int map[k];
    int pre_temp[nc];
    int pos;

    if(k!=1)
    {

        while((m=rem_nodes[i++])!=0){

            copy(var,rem_nodes,k);
            modify(var,k);

            copy(pre_temp,prev_nodes,nc);
            append(m, pre_temp,nc);

            map[ind]=m;
            temp[ind++]=c[node][m] + bellman(m,pre_temp,nc,c,var, k-1,o);

            clear(pre_temp,nc);
            clear(var,k);
        }

        r=min(temp,k,0);
        next =map[(int)min(temp,k,1)-1];

        pos = hash_lookup(prev_nodes,nc);
        o[node][pos] = next;

        return r;
    }
    else
    {
        return (c[node][rem_nodes[0]] + c[rem_nodes[0]][0]);

    }
}

void main()
{
    int n,i,j,temp=0;
    printf("Enter the number of cities: ");
    scanf("%d",&n);

    //Define lookup table. Map the remaining nodes array to an index and assign a value at that index ( hash value ).

    int limit= get_limit(n);
    int order[n][limit];

    //--------------------------------------------------------------------------

    float city[n][n];
    float opt;
    int prev[n];

    clear(prev,n);

    // Initialize the set of remaining nodes------------------------------------
    int set[n];
    for(i=0;i<n-1;i++)
    {
        set[i]=i+1;

    }
    set[n-1]=0;
    //--------------------------------------------------------------------------

    read(city,n);
    print_matrix(city,n);


    opt=bellman(0,prev,n,city,set,n-1,order);


    printf("\nOptimum Distance : %f\n", opt);

    // Print Optimum path-------------------------------------------------------

    int node=0;
    int next=0;
    int path[n];
    clear(path,n);

    printf("0");
    for(i=0;i<n-3;i++){

        printf("-->%d",order[node][next]);
        node=order[node][next];
        append(node,path,n);
        next=hash_lookup(path,n);

    }
}
     //-------------------------------------------------------------------------
