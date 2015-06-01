#include "default_defines.h"
#include "global_definitions.h"
#include "device.h"
#include "openmp.h"
#include "sotl.h"

#ifdef HAVE_LIBGL
#include "vbo.h"
#endif

#include <stdio.h>
#include <string.h>
#include <omp.h>

static int *atom_state = NULL;
static int *nb_elements = NULL;
static int *last_element = NULL;
static int *intermediate_sum = NULL;
static sotl_atom_set_t* tmp_set = NULL;


#define SHOCK_PERIOD  50

#ifdef HAVE_LIBGL

// Update OpenGL Vertex Buffer Object
//
static void omp_update_vbo (sotl_device_t *dev)
{
    sotl_atom_set_t *set = &dev->atom_set;
    //sotl_domain_t *domain = &dev->domain;

    for (unsigned n = 0; n < set->natoms; n++) {
        vbo_vertex[n*3 + 0] = set->pos.x[n];
        vbo_vertex[n*3 + 1] = set->pos.y[n];
        vbo_vertex[n*3 + 2] = set->pos.z[n];

        // Atom color depends on z coordinate
        {
            //float ratio = (set->pos.z[n] - domain->min_ext[2]) / (domain->max_ext[2] - domain->min_ext[2]);
            float ratio = atom_state[n]/(float)omp_get_num_procs();

            vbo_color[n*3 + 0] = (1.0 - ratio) * atom_color[0].R + ratio * 1.0;
            vbo_color[n*3 + 1] = (1.0 - ratio) * atom_color[0].G + ratio * 0.0;
            vbo_color[n*3 + 2] = (1.0 - ratio) * atom_color[0].B + ratio * 0.0;
            atom_state[n]--;
        }
    }
}
#endif

// Update positions of atoms by adding (dx, dy, dz)
//
static void omp_move (sotl_device_t *dev)
{
    sotl_atom_set_t *set = &dev->atom_set;

    for (unsigned n = 0; n < set->natoms; n++) {
        set->pos.x[n] += set->speed.dx[n];
        set->pos.y[n] += set->speed.dy[n];
        set->pos.z[n] += set->speed.dz[n];
    }
}

// Apply gravity force
//
static void omp_gravity (sotl_device_t *dev)
{
    sotl_atom_set_t *set = &dev->atom_set;
    const calc_t g = 0.005;

    //TODO
}

static void omp_bounce (sotl_device_t *dev)
{
    sotl_atom_set_t *set = &dev->atom_set;
    sotl_domain_t *domain = &dev->domain;

#pragma omp parallel for schedule(static)
    for (unsigned n = 0; n < set->natoms; n++) {
        if (set->pos.x[n]+set->speed.dx[n] <= domain->min_ext[0]){
            set->speed.dx[n] = -0.5*set-> speed.dx[n];
            atom_state[n]=SHOCK_PERIOD;
        }
        if (set->pos.x[n]+set->speed.dx[n] >= domain->max_ext[0]){
            set->speed.dx[n] = -0.5*set-> speed.dx[n];
            atom_state[n]=SHOCK_PERIOD;
        }
        if (set->pos.y[n]+set->speed.dy[n] <= domain->min_ext[1]){
            set->speed.dy[n] = -0.5*set-> speed.dy[n];
            atom_state[n]=SHOCK_PERIOD;
        }
        if (set->pos.y[n]+set->speed.dy[n] >= domain->max_ext[1]){
            set->speed.dy[n] = -0.5*set->speed.dy[n];
            atom_state[n]=SHOCK_PERIOD;
        }
        if (set->pos.z[n]+set->speed.dz[n] <= domain->min_ext[2]){
            set->speed.dz[n] = -0.5*set-> speed.dz[n];
            atom_state[n]=SHOCK_PERIOD;
        }
        if (set->pos.z[n]+set->speed.dz[n] >= domain->max_ext[2]){
            set->speed.dz[n] = -0.5*set-> speed.dz[n];
            atom_state[n]=SHOCK_PERIOD;
        }
    }
}

static calc_t squared_distance (sotl_atom_set_t *set, unsigned p1, unsigned p2)
{
    calc_t *pos1 = set->pos.x + p1,
           *pos2 = set->pos.x + p2;

    calc_t dx = pos2[0] - pos1[0],
           dy = pos2[set->offset] - pos1[set->offset],
           dz = pos2[set->offset*2] - pos1[set->offset*2];

    return dx * dx + dy * dy + dz * dz;
}

static calc_t lennard_jones (calc_t r2)
{
    calc_t rr2 = 1.0 / r2;
    calc_t r6;

    r6 = LENNARD_SIGMA * LENNARD_SIGMA * rr2;
    r6 = r6 * r6 * r6;

    return 24 * LENNARD_EPSILON * rr2 * (2.0f * r6 * r6 - r6);
}

static void set_force(sotl_atom_set_t *set, calc_t *force, int current, int other){
    if (current != other) {
        calc_t sq_dist = squared_distance (set, current, other);

        if (sq_dist < LENNARD_SQUARED_CUTOFF) {
            calc_t intensity = lennard_jones (sq_dist);

            force[0] += intensity * (set->pos.x[current] - set->pos.x[other]);
            force[1] += intensity * (set->pos.x[set->offset + current] -
                    set->pos.x[set->offset + other]);
            force[2] += intensity * (set->pos.x[set->offset * 2 + current] -
                    set->pos.x[set->offset * 2 + other]);
        }

    }

}

static void omp_force_z (sotl_device_t *dev)
{
    sotl_atom_set_t *set = &dev->atom_set;
    atom_set_sort(set);
    int min_z = 0;//this variable indicate the lower bound of atoms in the lennard_cutoff range for each thread
#pragma omp parallel for schedule(static) firstprivate(min_z)
    for (unsigned int current = 0; current < set->natoms; current++) {
        atom_state[current] = omp_get_thread_num();
        calc_t force[3] = { 0.0, 0.0, 0.0 };
    int *intermediate_sum = malloc(sizeof(int)*(omp_get_max_threads()+1));



        for (unsigned int other = min_z; other < set->natoms; other++){
            if (set->pos.z[other] - set->pos.z[current] > LENNARD_CUTOFF){ //Atoms not in the range (Max)
                break;
            }
            if (set->pos.z[current] - set->pos.z[other] > LENNARD_CUTOFF){ // Atoms not in the range (Min)
                min_z ++;
            }
            else{ //Atoms in the range

                //Set forces
                set_force(set, force, current, other);
            }
        }

        set->speed.dx[current] += force[0];
        set->speed.dx[set->offset + current] += force[1];
        set->speed.dx[set->offset * 2 + current] += force[2];
    }
}

static void omp_force_boites (sotl_device_t *dev){
    sotl_atom_set_t *set = &dev->atom_set;
    sotl_domain_t *domain = &dev->domain;


    memset(nb_elements, 0, sizeof(*nb_elements)*domain->total_boxes);

#pragma omp parallel for schedule(static)
    for (unsigned int current = 0; current < set->natoms; current ++){ //Define number of atom per Boxes
        int box_id = atom_get_num_box(domain, set->pos.x[current],
                set->pos.y[current], set->pos.z[current],BOX_SIZE_INV);
#pragma omp atomic
        nb_elements[box_id]++;
    }

    //caluclate cumulate sum.
    memset(intermediate_sum, 0, sizeof(int)*(omp_get_max_threads()+1));
    int split = domain->total_boxes/omp_get_max_threads();
#pragma omp parallel
    {
        for (unsigned int i = omp_get_thread_num()*split;
                i < (omp_get_thread_num()+1)*split;
                i++){
            intermediate_sum[omp_get_thread_num()+1]+=nb_elements[i];
        }
    }//remember that the last elements are not count
    //printf("total : %d\n",set->natoms);
    //printf("int[0]%d\n",intermediate_sum[0]);
    for (unsigned int i = 1; i < omp_get_max_threads(); i++){
        intermediate_sum[i]+=intermediate_sum[i-1];
        //printf("intermediate[%d]\t%d\n",i,intermediate_sum[i]);
    }
    for (unsigned int i = 0; i < omp_get_max_threads(); i++){
        last_element[i*split]=intermediate_sum[i];
    }
#pragma omp parallel
    {
        for (unsigned int i = omp_get_thread_num()*split+1;
                i < (omp_get_thread_num()+1)*split;
                i++){
            last_element[i] = last_element[i-1] + nb_elements[i];
        }
    }
    //we compute sum for last elements of the array
    for (unsigned int i = omp_get_max_threads()*split;
            i < (omp_get_thread_num()+1)*split;
            i++){
        last_element[i] = last_element[i-1] + nb_elements[i];
    }
    last_element[domain->total_boxes] = set->natoms;


#pragma omp parallel for schedule(static)
    for (unsigned int current = 0; current < set->natoms; current ++){//Place all atoms in related box
        int box_id = atom_get_num_box(domain, set->pos.x[current],
                set->pos.y[current], set->pos.z[current],BOX_SIZE_INV);
        unsigned int new_pos;
#pragma omp atomic capture
        new_pos = --last_element[box_id];
        //printf("%d\n",new_pos);
        tmp_set->pos.x[new_pos] = set->pos.x[current];
        tmp_set->pos.y[new_pos] = set->pos.y[current];
        tmp_set->pos.z[new_pos] = set->pos.z[current];
        tmp_set->speed.dx[new_pos] = set->speed.dx[current];
        tmp_set->speed.dy[new_pos] = set->speed.dy[current];
        tmp_set->speed.dz[new_pos] = set->speed.dz[current];
    }
    //switch the two sets
    sotl_atom_set_t old_tmp_set = *tmp_set;
    *tmp_set = *set;
    *set = old_tmp_set;

    //Now compute interractions
#pragma omp parallel for schedule(static)
    for (unsigned int current = 0; current < set->natoms; current++) {
        calc_t force[3] = { 0.0, 0.0, 0.0 };
        atom_state[current] = omp_get_thread_num();

        int box_id = atom_get_num_box(domain, set->pos.x[current], set->pos.y[current], set->pos.z[current],BOX_SIZE_INV);
        for (int i=-1; i<=1; i++){
            for (int j=-1; j<=1; j++){
                for (int k=-1; k<=1; k++){
                    int other_box_id = box_id+
                        i+
                        (j*domain->boxes[0])+
                        (k*domain->boxes[0]*domain->boxes[1]);
                    if (other_box_id < 0 || (unsigned) other_box_id > domain->total_boxes)
                        continue;
                    for (unsigned int other = last_element[other_box_id];
                            other < last_element[other_box_id+1];
                            other++){
                        set_force(set, force, current, other);
                    }
                }
            }
        }
        set->speed.dx[current] += force[0];
        set->speed.dy[current] += force[1];
        set->speed.dz[current] += force[2];
    }



    /*
       for (unsigned int other = min_z; other < set->natoms; other++){
       if (set->pos.z[other] - set->pos.z[current] > LENNARD_CUTOFF){ //Atoms not in the range (Max)
       break;
       }
       if (set->pos.z[current] - set->pos.z[other] > LENNARD_CUTOFF){ // Atoms not in the range (Min)
       min_z ++;
       }
       else{ //Atoms in the range

    //Set forces
    set_force(set, force, current, other);
    }
    }

    set->speed.dx[current] += force[0];
    set->speed.dx[set->offset + current] += force[1];
    set->speed.dx[set->offset * 2 + current] += force[2];
    }
    */
}

static void omp_force (sotl_device_t *dev){
    omp_force_boites(dev);
}

// Main simulation function
//
void omp_one_step_move (sotl_device_t *dev)
{
    // Apply gravity force
    //
    if (gravity_enabled)
        omp_gravity (dev);

    // Compute interactions between atoms
    //
    if (force_enabled)
        omp_force (dev);

    // Bounce on borders
    //
    if(borders_enabled)
        omp_bounce (dev);

    // Update positions
    //
    omp_move (dev);

#ifdef HAVE_LIBGL
    // Update OpenGL position
    //
    if (dev->display)
        omp_update_vbo (dev);
#endif
}

void omp_init (sotl_device_t *dev)
{
#ifdef _SPHERE_MODE_
    sotl_log(ERROR, "Sequential implementation does currently not support SPHERE_MODE\n");
    exit (1);
#endif


    borders_enabled = 1;

    dev->compute = SOTL_COMPUTE_OMP; // dummy op to avoid warning
}

void omp_alloc_buffers (sotl_device_t *dev)
{
    tmp_set = malloc(sizeof(*tmp_set));
    atom_set_init(tmp_set,dev->atom_set.natoms,dev->atom_set.natoms);
    atom_state = calloc(dev->atom_set.natoms, sizeof(int));
    printf("natoms: %d\n", dev->atom_set.natoms);
    nb_elements = malloc(sizeof(*nb_elements)*dev->domain.total_boxes);
    last_element = malloc(sizeof(*last_element)*(dev->domain.total_boxes + 1));
    intermediate_sum = malloc(sizeof(int)*(omp_get_max_threads()+1));
}

void omp_finalize (sotl_device_t *dev)
{
    free(atom_state);
    atom_set_free(tmp_set);
    dev->compute = SOTL_COMPUTE_OMP; // dummy op to avoid warning
    free(nb_elements);
    free(last_element);
    free(intermediate_sum);
}
