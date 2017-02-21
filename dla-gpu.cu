// **************************************************************************************************
// *** IMPORTANT INFORMATION ************************************************************************
// **************************************************************************************************

// - Installation guide: https://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/
// - To compile samples, set permissions to "Read & Write" for folder and enclosed items
// - See Makefile for custom compilation process
// - Xcode version reverted to 6.2 (last supported version, https://developer.apple.com/downloads/)


// **************************************************************************************************
// *** INCLUDES AND DEFINES *************************************************************************
// **************************************************************************************************


#include <fstream>
#include <sstream>

#include <thrust/version.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/functional.h>
#include <thrust/set_operations.h>
#include <thrust/unique.h>
#include <thrust/extrema.h>

#define BC_BORDERS 0
#define BC_NUMBER 1
#define X 0
#define Y 1
#define Z 2


// **************************************************************************************************
// *** NAMESPACES USED ******************************************************************************
// **************************************************************************************************


using namespace std;
using namespace thrust::placeholders;


// **************************************************************************************************
// *** SETTINGS *************************************************************************************
// **************************************************************************************************


// Internal parameters
const int MEM_MAX = 10000; // Maximum path values to be examined in parallel (total allocated size in bytes is ca. 5*MEM_MAX)
const int SPAWN_DIST = 10; // Spawning distance
const int PATH_QUOT = 100; // Each examined path will be PATH_QUOT times longer than the spawning distance
const int PATH_LENGTH = SPAWN_DIST * PATH_QUOT;
const int PATH_COUNT = MEM_MAX / PATH_LENGTH;
const int TOTAL_LENGTH = PATH_COUNT * PATH_LENGTH;

// Settings for pyrit
/*
const int DIM = 100;                                // Grid size is DIM*DIM*DIM
const int LEFT_PROB_BIG = 30;                       // Probability of big cubes appearing in the left half
const int RIGHT_PROB_BIG = 2;                       // Probability of big cubes appearing in the right half
const int DELIM = DIM/2;                            // Where is the border between the two "halfs" of the grid along the z axis?
const int BIG_MIN = 100;                            // Minimum size for big cube
const int BIG_MAX = 150;                            // Maximum size for big cube
const int BIG_REP = 1;                              // Number of in-place rotations for big cube
const int SMALL_MIN = 80;                           // Minimum size for small cube
const int SMALL_MAX = 120;                          // Maximum size for small cube
const int SMALL_REP = 20;                           // Number of in-place rotations for small cube
const int BC_NUMBER_VAL = DIM*DIM;                  // Setting for breaking condition BC_NUMBER
const bool breaking_condition = BC_NUMBER;          // Type of breaking condition: BC_BORDERS or BC_NUMBER
const string ROCKFILE = "rock1.obj";                // Name of the host structure file, leave empty to just start with the center
const string OUTFILE = "pyrit.obj";                 // Name of the output file
const float HOST_SIZE = 2;                          // Host structure size (1/HOST_SIZE) in regard to grid size
*/

// Settings for coral
// /*
const int DIM = 500;                                // Grid size is DIM*DIM*DIM
const int LEFT_PROB_BIG = 0;                        // Probability of big cubes appearing in the left half
const int RIGHT_PROB_BIG = 0;                       // Probability of big cubes appearing in the right half
const int DELIM = DIM/2;                            // Where is the border between the two "halfs" of the grid along the z axis?
const int BIG_MIN = 100;                            // Minimum size for big cube
const int BIG_MAX = 150;                            // Maximum size for big cube
const int BIG_REP = 1;                              // Number of in-place rotations for big cube
const int SMALL_MIN = 80;                           // Minimum size for small cube
const int SMALL_MAX = 120;                          // Maximum size for small cube
const int SMALL_REP = 20;                           // Number of in-place rotations for small cube
const int BC_NUMBER_VAL = DIM*DIM;                  // Setting for breaking condition BC_NUMBER
const bool breaking_condition = BC_BORDERS;         // Type of breaking condition: BC_BORDERS or BC_NUMBER
const string ROCKFILE = "rock2.obj";                // Name of the host structure file, leave empty to just start with the center
const string OUTFILE = "coral.obj";                 // Name of the output file
const float HOST_SIZE = 8;                          // Host structure size (1/HOST_SIZE) in regard to grid size
// */


// **************************************************************************************************
// *** GLOBALS AND TYPEDEFS *************************************************************************
// **************************************************************************************************


stringstream faces_stream;
stringstream vertices_stream;
ofstream out;

// https://github.com/thrust/thrust/blob/master/examples/dot_products_with_zip.cu
typedef thrust::tuple<int, int, int, int> tp;
typedef thrust::tuple<float, float, float, float> tp_f;
typedef thrust::device_vector<int>::iterator IntIterator;
typedef thrust::tuple<IntIterator, IntIterator, IntIterator, IntIterator> IntIteratorTuple;
typedef thrust::zip_iterator<IntIteratorTuple> tp_it;

// Points of the host structure
thrust::host_vector<tp> obj_points;
int obj_points_size = 0;


// **************************************************************************************************
// *** SORTING FUNCTIONS ****************************************************************************
// **************************************************************************************************


// Tuple sorting on the device, has to adhere to "strict weak ordering" (false when equal!)
// http://stackoverflow.com/questions/21075778/how-does-thrust-set-intersection-works
// http://www.sgi.com/tech/stl/StrictWeakOrdering.html
struct sortXYZ {
    __device__
    int operator()(const tp a, const tp b) const {

        int x1 = thrust::get<1>(a); int y1 = thrust::get<2>(a); int z1 = thrust::get<3>(a);
        int x2 = thrust::get<1>(b); int y2 = thrust::get<2>(b); int z2 = thrust::get<3>(b);

        return  x1 == x2 ? (y1 == y2 ? (z1 == z2 ? false : z1 < z2) : y1 < y2) : x1 < x2;
    }
};


// Returns the lowest of 2*3 tuple parts, on the device
struct lbXYZ2 {
    __device__
    int operator()(const tp a, const tp b) const {

        int x1 = thrust::get<1>(a); int y1 = thrust::get<2>(a); int z1 = thrust::get<3>(a);
        int x2 = thrust::get<1>(b); int y2 = thrust::get<2>(b); int z2 = thrust::get<3>(b);

        int lb_a = x1 < y1 ? (x1 < z1 ? x1 : z1) : (y1 < z1 ? y1 : z1);
        int lb_b = x2 < y2 ? (x2 < z2 ? x2 : z2) : (y2 < z2 ? y2 : z2);

        return lb_a < lb_b ;
    }
};


// Returns the lowest of 3 tuple parts
__host__
int lbXYZ1(const tp &a) {
    int x = thrust::get<1>(a); int y = thrust::get<2>(a); int z = thrust::get<3>(a);
    return x < y ? (x < z ? x : z) : (y < z ? y : z);
}


// Returns the highest of 3 tuple parts
__host__
int ubXYZ1(const tp &a) {
    int x = thrust::get<1>(a); int y = thrust::get<2>(a); int z = thrust::get<3>(a);
    return x > y ? (x > z ? x : z) : (y > z ? y : z);
}


// Index comparison on the device
struct compI
{
    __device__
    int operator()(const tp a, const tp b) const {

        int i1 = thrust::get<0>(a);
        int i2 = thrust::get<0>(b);

        return i1 < i2;
    }
};


// **************************************************************************************************
// *** 3D FUNCTIONS *********************************************************************************
// **************************************************************************************************


// http://en.wikipedia.org/wiki/Rotation_matrix
// http://en.wikipedia.org/wiki/Matrix_multiplication
// http://inside.mines.edu/fs_home/gmurray/ArbitraryAxisRotation/
void rotate(vector<float> &pV, vector<float> &pO, float theta, int d) {

    // Rotation of (x,y,z) around the line through (a,b,c) with direction unit vector (u,v,w)
    float a = pO[0];
    float b = pO[1];
    float c = pO[2];

    float x = pV[0];
    float y = pV[1];
    float z = pV[2];

    float u,v,w;

    if (d == X) {u = 1;} else {u=0;}
    if (d == Y) {v = 1;} else {v=0;}
    if (d == Z) {w = 1;} else {w=0;}

    float x_new = (a * (v*v + w*w) - u * (b*v + c*w -u*x -v*y -w*z)) * (1 - cos(theta)) + x * cos(theta) + (-c*v + b*w - w*y + v*z) * sin(theta);
    float y_new = (b * (u*u + w*w) - v * (a*u + c*w -u*x -v*y -w*z)) * (1 - cos(theta)) + y * cos(theta) + (c*u - a*w + w*x - u*z) * sin(theta);
    float z_new = (c * (u*u + v*v) - w * (a*u + b*v -u*x -v*y -w*z)) * (1 - cos(theta)) + z * cos(theta) + (-b*u + a*v - v*x + u*y) * sin(theta);

    pV[0] = x_new;
    pV[1] = y_new;
    pV[2] = z_new;

}


// **************************************************************************************************
// *** OBJ INPUT AND OUTPUT *************************************************************************
// **************************************************************************************************


void readHostStructure(string filename) {

    static float max = 0.0;
    static float min = 0.0;
    static float mm;
    thrust::host_vector<tp_f> temp_points;

    ifstream infile(filename);
    string s;
    while (getline(infile, s)) {
        stringstream ss(s);
        vector<string> line;
        string item;
        while (getline(ss, item, ' ')) {
            line.push_back(item);
        }
        if (line[0] == "v") {

            // Read the coordinates
            float x = stof(line[1]);
            float y = stof(line[2]);
            float z = stof(line[3]);

            // Find minimum and maximum
            if (x > max) {max = x;}
            if (y > max) {max = y;}
            if (z > max) {max = z;}
            if (x < min) {min = x;}
            if (y < min) {min = y;}
            if (z < min) {min = z;}

            // Write to temporary vector
            temp_points.push_back(tp_f(0,x,y,z));

        }

    }

    if (max > abs(min)) {mm = max;}
    else {mm = abs(min);}

    for (tp_f temp : temp_points) {

            // Read values back in
            float x = thrust::get<1>(temp);
            float y = thrust::get<2>(temp);
            float z = thrust::get<3>(temp);

            // Scale from original value range to -1 / 1 range
            x /= mm;
            y /= mm;
            z /= mm;

            // Scale complete object, account for -1 / 1 range (*2)
            x *= (DIM/(HOST_SIZE*2));
            y *= (DIM/(HOST_SIZE*2));
            z *= (DIM/(HOST_SIZE*2));

            // Translate object to middle of the grid
            x += DIM/2;
            y += DIM/2;
            z += DIM/2;

            // Round to nearest integer (sampling the object)
            int ix = int(x);
            int iy = int(y);
            int iz = int(z);

            // Write to final vector
            obj_points.push_back(tp(0,ix,iy,iz));

    }

    // Remove duplicates from obj_points in place (sampling artifacts)
    thrust::sort(obj_points.begin(), obj_points.end(), sortXYZ());
    cout << "Original host OBJ has " << obj_points.size() << " vertices. " << endl;
    auto new_end_unique = thrust::unique(obj_points.begin(), obj_points.end());
    obj_points.resize(thrust::distance(obj_points.begin(), new_end_unique));
    cout << "Cleaned up host OBJ has " << obj_points.size() << " vertices. " << endl;
    obj_points_size = obj_points.size();

}


void pointsToCubes(thrust::host_vector<tp> &pPoints) {

    // We may call the function several times if we want
    static int index = 0;

    for (tp point : pPoints) {

        // Redefinition for convenience
        float x = thrust::get<1>(point);
        float y = thrust::get<2>(point);
        float z = thrust::get<3>(point);

        // Initialize randomness
        int n = rand()%100;
        float r; int m;

        // One half along the z axis
        if (z<DELIM) {
            if (n<LEFT_PROB_BIG) {
                float nn = (rand()%(BIG_MAX-BIG_MIN))+BIG_MIN;
                nn /= 100.0;
                r=nn;
                m=BIG_REP;}
            else {
                float nn = (rand()%(SMALL_MAX-SMALL_MIN))+SMALL_MIN;
                nn /= 100.0;
                r=nn;
                m=SMALL_REP;}
        }

        // Other half along the z axis
        else {
            if (n<RIGHT_PROB_BIG) {
                float nn = (rand()%(BIG_MAX-BIG_MIN))+BIG_MIN;
                nn /= 100.0;
                r=nn;
                m=BIG_REP;}
            else {
                float nn = (rand()%(SMALL_MAX-SMALL_MIN))+SMALL_MIN;
                nn /= 100.0;
                r=nn;
                m=SMALL_REP;}
        }

        for (int i=0; i<m; i++) {

            vector<vector <float>>  vertices = {
                {x-r,y+r,z+r}, // 1: top left, back
                {x+r,y+r,z+r}, // 2: top right, back
                {x+r,y-r,z+r}, // 3: bottom right, back
                {x-r,y-r,z+r}, // 4: bottom left, back
                {x-r,y+r,z-r}, // 5: top left, front
                {x+r,y+r,z-r}, // 6: top right, front
                {x+r,y-r,z-r}, // 7: bottom right, front
                {x-r,y-r,z-r}  // 8: bottom left, front
            };

            vector<vector <int>> faces = {
                {1,2,3}, {4,1,3}, // back
                {5,6,7}, {8,5,7}, // front
                {1,5,2}, {6,2,5}, // top
                {4,3,8}, {7,8,3}, // bottom
                {1,4,5}, {8,5,4}, // left
                {2,3,6}, {7,6,3}  // right
            };

            float thetaX = rand()%360;
            float thetaY = rand()%360;
            float thetaZ = rand()%360;


            for (vector<float> &v : vertices) {

                // Rotate  randomly around the original vertex point
                vector<float> o = {x,y,z};
                rotate(v, o, thetaX, X);
                rotate(v, o, thetaY, Y);
                rotate(v, o, thetaZ, Z);

                // Normalize in regard to grid dimensions
                v[0] = (v[0]-DIM/2)/DIM;
                v[1] = (v[1]-DIM/2)/DIM;
                v[2] = (v[2]-DIM/2)/DIM;

                // Write to vertex stream
                vertices_stream << "v " << v[0] << " " << v[1] << " " << v[2] << " 1.0 \n";

            }

            for (vector<int> &f : faces) {
                faces_stream << "f";
                for (int &i : f) {faces_stream << " " << i+index;}
                faces_stream << "\n";
            }

            index += vertices.size();

        }

    }

}


// **************************************************************************************************
// *** GPU HELPER FUNCTIONS *************************************************************************
// **************************************************************************************************


void push_docking(thrust::device_vector<tp> &pDocking_points, const tp &pTp) {
    // Add all six neighbours, index is not used but must exist for comparison sanity, thus set to zero
    pDocking_points.push_back(thrust::make_tuple(0,thrust::get<1>(pTp), thrust::get<2>(pTp), thrust::get<3>(pTp)+1));
    pDocking_points.push_back(thrust::make_tuple(0,thrust::get<1>(pTp), thrust::get<2>(pTp), thrust::get<3>(pTp)-1));
    pDocking_points.push_back(thrust::make_tuple(0,thrust::get<1>(pTp), thrust::get<2>(pTp)+1, thrust::get<3>(pTp)));
    pDocking_points.push_back(thrust::make_tuple(0,thrust::get<1>(pTp), thrust::get<2>(pTp)-1, thrust::get<3>(pTp)));
    pDocking_points.push_back(thrust::make_tuple(0,thrust::get<1>(pTp)+1, thrust::get<2>(pTp), thrust::get<3>(pTp)));
    pDocking_points.push_back(thrust::make_tuple(0,thrust::get<1>(pTp)-1, thrust::get<2>(pTp), thrust::get<3>(pTp)));
}


void push_frozen(thrust::device_vector<tp> &pFrozen_points, const tp &pTp) {
    // Index is not used but must exist for comparison sanity, thus set to zero
    pFrozen_points.push_back(thrust::make_tuple(0, thrust::get<1>(pTp), thrust::get<2>(pTp), thrust::get<3>(pTp)));
}


void clean(thrust::device_vector<tp> &pDocking_points, thrust::device_vector<tp> &pFrozen_points) {
    // Sort everything as a prerequisite for the next two steps
    thrust::sort(pDocking_points.begin(), pDocking_points.end(), sortXYZ());
    thrust::sort(pFrozen_points.begin(), pFrozen_points.end(), sortXYZ());

    // Remove duplicates from docking_points in place
    auto new_end_unique = thrust::unique(pDocking_points.begin(), pDocking_points.end());
    pDocking_points.resize(thrust::distance(pDocking_points.begin(), new_end_unique));

    // Remove frozen_points from docking_points
    auto new_end_set_difference = thrust::set_difference(pDocking_points.begin(), pDocking_points.end(), pFrozen_points.begin(), pFrozen_points.end(), pDocking_points.begin(), sortXYZ());
    pDocking_points.resize(thrust::distance(pDocking_points.begin(),new_end_set_difference));

    // Sort docking_points again so there's no need to do it every try
    thrust::sort(pDocking_points.begin(), pDocking_points.end(), sortXYZ());
}


// **************************************************************************************************
// *** MAIN *****************************************************************************************
// **************************************************************************************************


// Why this can not be a __device__ function (NEW: Thrust 1.8 allows it!)
// http://stackoverflow.com/questions/20157815/how-to-access-a-device-vector-from-a-functor
// LONG-TERM TO DO: Use structures of arrays rather than arrays of structures everywhere!
// LONG-TERM TO DO: Implement compressed sorting of tuples: http://people.maths.ox.ac.uk/gilesm/cuda/lecs/thrust.pdf
int main(int argc, char **argv) {

    // Seed the host"s rng
    srand(time(NULL));

    if (ROCKFILE != "") {
        // Read in host structure
        readHostStructure(ROCKFILE);
        cout << "Reading host OBJ..." << endl;
    }

    // Open output file for writing
    out.open (OUTFILE);
    cout << "Out OBJ opened for writing." << "\n";

    // Allocate vectors
    thrust::device_vector<tp> docking_points;
    thrust::device_vector<tp> frozen_points;

    if (ROCKFILE != "") {
        // Add points from host structure
        for (tp &point : obj_points) {
            push_docking(docking_points, point);
            push_frozen(frozen_points, point);
        }
        clean(docking_points, frozen_points);
    }

    else {
        tp point = tp(0,DIM/2,DIM/2,DIM/2);
        push_docking(docking_points, point);
        push_frozen(frozen_points, point);
        clean(docking_points, frozen_points);
    }

    int min_dist = DIM/2;
    int spawning_dist = min_dist - SPAWN_DIST;
    bool done = false;
    int tries = 0;

    // Allocate device vectors, will be recycled
    thrust::device_vector<int> indexPath(TOTAL_LENGTH);
    thrust::device_vector<int> keys(TOTAL_LENGTH);
    thrust::device_vector<int> xPath(TOTAL_LENGTH);
    thrust::device_vector<int> yPath(TOTAL_LENGTH);
    thrust::device_vector<int> zPath(TOTAL_LENGTH);

    // Allocate host vectors, will be recycled
    thrust::host_vector<int> h_xPath(TOTAL_LENGTH);
    thrust::host_vector<int> h_yPath(TOTAL_LENGTH);
    thrust::host_vector<int> h_zPath(TOTAL_LENGTH);


    // **************************************************************************************************
    // *** MAIN GPU LOOP ********************************************************************************
    // **************************************************************************************************


    // LONG-TERM TO DO: This loop should run completely "on the device" until a point is found
    while(!done) {

        tries++;

        // Fill host vectors with random numbers
        // LONG-TERM TO DO: Random number generator on the device has not worked, regardless of adhering to:
        // http://stackoverflow.com/questions/12614164/generating-a-random-number-vector-between-0-and-1-0-using-thrust
        // http://stackoverflow.com/questions/19023070/why-is-thrust-uniform-random-distribution-generating-wrong-values
        // https://code.google.com/p/thrust/source/browse/examples/monte_carlo.cu
        thrust::generate(h_xPath.begin(), h_xPath.end(), rand);
        thrust::generate(h_yPath.begin(), h_yPath.end(), rand);
        thrust::generate(h_zPath.begin(), h_zPath.end(), rand);

        // Copy
        xPath = h_xPath;
        yPath = h_yPath;
        zPath = h_zPath;

        // Find starting side of the cube
        int lb = spawning_dist;
        int ub = DIM-((2 * spawning_dist));
        int select = rand()%6;
        int r1 = (rand()%ub)+lb;
        int r2 = (rand()%ub)+lb;
        int xStart, yStart, zStart;

        if (select == 0) {xStart=spawning_dist; yStart=r1; zStart=r2;}         // left
        if (select == 1) {xStart=DIM-(spawning_dist+1); yStart=r1; zStart=r2;} // right
        if (select == 2) {xStart=r1; yStart=spawning_dist; zStart=r2;}         // down
        if (select == 3) {xStart=r1; yStart=DIM-(spawning_dist+1); zStart=r2;} // up
        if (select == 4) {xStart=r1; yStart=r2; zStart=spawning_dist;}         // near
        if (select == 5) {xStart=r1; yStart=r2; zStart=DIM-(spawning_dist+1);} // far

        // Transform random numbers into steps
        thrust::transform(xPath.begin(), xPath.end(), xPath.begin(), (_1%3)-1);
        thrust::transform(yPath.begin(), yPath.end(), yPath.begin(), (_1%3)-1);
        thrust::transform(zPath.begin(), zPath.end(), zPath.begin(), (_1%3)-1);

        // Fill index vector with sequence
        thrust::sequence(indexPath.begin(), indexPath.end());

        // Fill key vector with a PATH_COUNT part sequence of PATH_LENGTH long items (using placeholders to avoid functors)
        // Why are we doing this: https://groups.google.com/forum/#!topic/thrust-users/RXm78Xl2waU
        thrust::tabulate(keys.begin(), keys.end(), _1/PATH_LENGTH);

        // Prefix sum calculation in place
        thrust::exclusive_scan_by_key(keys.begin(), keys.end(), xPath.begin(), xPath.begin(), xStart);
        thrust::exclusive_scan_by_key(keys.begin(), keys.end(), yPath.begin(), yPath.begin(), yStart);
        thrust::exclusive_scan_by_key(keys.begin(), keys.end(), zPath.begin(), zPath.begin(), zStart);

        // Zip together with index
        tp_it path_points_first = thrust::make_zip_iterator(thrust::make_tuple(indexPath.begin(), xPath.begin(), yPath.begin(), zPath.begin()));
        tp_it path_points_last = thrust::make_zip_iterator(thrust::make_tuple(indexPath.end(), xPath.end(), yPath.end(), zPath.end()));

        // Upper bound: Number of matching points can be the number of points in a path max.
        thrust::device_vector<tp> matching_points(TOTAL_LENGTH);

        // Sort the virtual vector path_points, docking_points is always already sorted
        thrust::sort(path_points_first, path_points_last, sortXYZ());

        // Find all points that are both in path_points and docking_points and put them in matching_points: this is a LCS / string matching problem
        auto new_end_set_intersection = thrust::set_intersection(path_points_first, path_points_last, docking_points.begin(), docking_points.end(), matching_points.begin(), sortXYZ());
        matching_points.resize(thrust::distance(matching_points.begin(), new_end_set_intersection));


        // **************************************************************************************************
        // *** IF POINT IS FOUND ****************************************************************************
        // **************************************************************************************************


        if (matching_points.size() > 0) {

            // Find first matching point according to indices
            auto min_matching_point_it = thrust::min_element(matching_points.begin(), matching_points.end(), compI());
            tp min_matching_point = *min_matching_point_it;

            // Display info (dereferencing is ok because this only happens when points are found)
            int i = thrust::get<0>(min_matching_point);
            int x = thrust::get<1>(min_matching_point);
            int y = thrust::get<2>(min_matching_point);
            int z = thrust::get<3>(min_matching_point);
            cout << "MATCH: " << x << "/" << y << "/" << z << " WITH TOTAL INDEX " << (TOTAL_LENGTH * tries) + i << ", ";

            // Add first matching point (functions replace index with 0)
            push_docking(docking_points, min_matching_point);
            push_frozen(frozen_points, min_matching_point);
            clean(docking_points, frozen_points);

            auto min_frozen_point_it = thrust::min_element(frozen_points.begin(), frozen_points.end(), lbXYZ2());
            tp min_frozen_point = *min_frozen_point_it;
            int minimum = lbXYZ1(min_frozen_point);

            auto max_frozen_point_it = thrust::max_element(frozen_points.begin(), frozen_points.end(), lbXYZ2());
            tp max_frozen_point = *max_frozen_point_it;
            int maximum = ubXYZ1(max_frozen_point);

            // The point with the smallest distance to the borders is either next to 0 or next to DIM
            if (0+minimum <= DIM-maximum) {min_dist=minimum;}
            if (0+minimum > DIM-maximum) {min_dist=DIM-maximum;}

            spawning_dist = min_dist - SPAWN_DIST;
            if (spawning_dist < 0) {spawning_dist = 0;}

            // Examine breaking conditions
            if (breaking_condition == BC_BORDERS) {
                float progress = (DIM-min_dist)*(100.0/float(DIM));
                cout << "PROGRESS: " << (progress-50)*2 << "%." << endl;
                done = (min_dist == 0);
            }

            if (breaking_condition == BC_NUMBER) {
                cout << "PROGRESS: " << 100*(frozen_points.size()/float(BC_NUMBER_VAL)) << "%." << endl;
                // We might add several frozen points at once, so we might not reach the exact number
                done = (frozen_points.size() >= BC_NUMBER_VAL);
            }

            else {done = (min_dist == 0);}

            // Reset counter
            tries = 0;

        }

    }


    // **************************************************************************************************
    // *** MAIN CLEANUP *********************************************************************************
    // **************************************************************************************************


    // Collect final points on host
    thrust::host_vector<tp> final_points;
    final_points = frozen_points;

    cout << "TOTAL NUMBER OF POINTS: " << final_points.size() << endl;

    pointsToCubes(final_points);

    // Write to output file and close
    out << vertices_stream.str();
    out << faces_stream.str();
    out.close();
    cout << "Out OBJ closed." << "\n";

    exit(0);

}
