
#define max 6
#define max2 1956

void print_arr(int ar[],int l)
{
    int r;
    printf("\n");
    for(r=l-1;r>1;r--)
    {
        printf(" %d ",ar[r]);
    }
    printf("\n");
}
void read(float r[max][max], int n)
{
    int i,j;
    printf("Enter the distances from each city\n\n");

    for(i=0;i<n-1;i++)
    {
        printf("Enter the distances from city %d\n", i);
        for(j=i;j<n;j++)
        {
            if(i==j)
                r[i][j]=0;
            else{

                //scanf("%f",&r[i][j]);
                r[i][j]=i+j;
                r[j][i]=r[i][j];
            }

        }
    }
    r[n-1][n-1]=0;
}

void print_matrix(float m[max][max], int l)
{
    printf("\nThe Adjacency Matrix :\n");
    int i,j;
    for(i=0;i<l;i++)
    {
        for(j=0;j<l;j++)
        {
            printf(" %f ",m[i][j]);
        }
        printf("\n");
    }
}
