void kernel floyd(global int* graphe, //matrice adjacence
    global int* n,   //taille de la matrice
    global int* distances,  //matrice distance
    global int* k)   //index
{
    int op1;
    int op2;
    int i = get_global_id(0);
    int j = get_global_id(1);
    int o = i * (*n) + j;
    op1 = i * (*n) + (*k);
    op2 = (*k) * (*n) + j;
    if (graphe[o] < graphe[op1] + graphe[op2])
    {
        distances[o] = graphe[o];
    }
    else
    {
        distances[o] = graphe[op1] + graphe[op2];
    }
}