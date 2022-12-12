void kernel floyd(global int* graphe, //matrice adjacence
    global int* n,   //taille de la matrice
    global int* distances,  //matrice distance
    global int* k)   //index
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    int ij = i * (*n) + j;
    int ik = i * (*n) + (*k);
    int kj = (*k) * (*n) + j;

    if (distances[ik] + distances[kj] < distances[ij]) {
        distances[ij] = distances[ik] + distances[kj];
    }
}