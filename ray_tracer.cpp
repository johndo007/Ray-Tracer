/*
//--------------------------------------------------------------------------------
//Basic Idea And Resources
//--------------------------------------------------------------------------------
//Ray Tracing
for each pixel do
  compute viewing ray 
  if( ray hits an object with t in [0, inf] ) then
    compute normal 
    evaluate shading model and set pixel to that color 
  else
    set pixel color to the background color

//Phong Reflection Model
- l to light source
- n surface normal
- v to viewer
- r perfect reflector (function of n and l)

//Formula must be applied per color
I = Ia + Id + Is = Ra * La + Rd * Ld * max(0, 1 * n) + Rs * Ls * pow(max(0, cos(theta)), alpha)
I = Color Intensity
R = Reflectance
L = Illumination

//Shadows
for each pixel do
  compute viewing ray 
  if(ray hits an object with t in [0, inf]) then
      compute normal
      evaluate shading model and set pixel to that color 
  else
    set pixel color to the background color

//Reflections    
for each pixel do
  compute viewing ray 
  if(ray hits an object with t in [0, inf]) then
    compute normal 
    evaluate shading model and set pixel to that color 
  else
    set pixel color to the background color
    
http://www.cs.ucr.edu/~shinar/courses/cs130-fall-2015/lectures/Lecture8.pdf
http://www.cs.ucr.edu/~shinar/courses/cs130-fall-2015/lectures/Lecture12.pdf
http://www.cs.ucr.edu/~shinar/courses/cs130-fall-2015/lectures/Lecture13.pdf
https://www.cs.princeton.edu/courses/archive/fall00/cs426/lectures/raycast/sld001.htm
http://mrl.nyu.edu/~dzorin/cg05/lecture12.pdf
http://paulbourke.net/geometry/reflected/
http://www.cs.rpi.edu/~cutler/classes/advancedgraphics/F05/lectures/13_ray_tracing.pdf

//--------------------------------------------------------------------------------
//Basic Understanding of Ray Tracing and Ray-Sphere Intersection
//-------------------------------------------------------------------------------- 
Lab 7 Basic Ray Tracer
//Main Function
int main(int nArgs, char** args)
{
  //Initialize the window
  SDL_Init(SDL_INIT_VIDEO);
  SDL_InitSubSystem(SDL_INIT_VIDEO);

  SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8); //8 bits for red
  SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8); //8 bits for green
  SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8); //8 bits for blue
  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1); //enable page flipping
  SDL_SetVideoMode(SCREEN_WIDTH, SCREEN_HEIGHT, 32,SDL_OPENGL);
  SDL_WM_SetCaption("CS130 Lab", NULL);

  //Set up the projection - don't worry about this
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(XMIN, XMAX, YMIN, YMAX,-1,1);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  vector3 spherePosition(0,0,30);
  double sphereRadius=10;
  vector3 lightPosition(-20,-10,-10);
  vector3 pixelGridCenter(0,0,5);
  vector3 cameraPos(0,0,0);
  vector3 ambientColor(.2f,.2f,.2f);
  vector3 diffuseColor(.5f,.5f,.5f);
  vector3 specularColor(.5f,.5f,.5f);
  vector3 x_incr(.025,0,0);
  vector3 y_incr(0,.025,0);
  
  float c1 = 1, c2 = 1, c3 = 1;
  
  while(true)
  {
      //Update platform specific stuff
      SDL_Event event;
      SDL_PollEvent(&event); //Get events
      if(event.type == SDL_QUIT) //if you click the [X] exit out
          break;
      //Graphics
      glClear(GL_COLOR_BUFFER_BIT); //Clear the screen       
      for(int x = XMIN; x <= XMAX; ++x)
      {
          for(int y = YMIN; y <= YMAX; ++y)
          {
              //spacial position of pixelVector
              vector3 pixelVector = pixelGridCenter + x * x_incr + y * y_incr; 
              
              //u*
              vector3 cameraVector = pixelVector - cameraPos;
              cameraVector = normal(cameraVector);
              
              //Location of sphereVector pixelVector with respect to cameraVector
              vector3 sphereVector = cameraPos -spherePosition;
              
              //Radius of sphereVector squared for formula usage
              double r2 = pow(sphereRadius, 2.0);
              
              double negB = (-1) * (dot(sphereVector, cameraVector));
              double discriminant = sqrt(pow(dot(sphereVector, cameraVector), 2.0) - dot(sphereVector, sphereVector) + r2);
              
              double intersectionOne = negB - discriminant;
              double intersectionTwo = negB + discriminant;

              double delta =  pow(dot(sphereVector, cameraVector), 2.0) - dot(sphereVector, sphereVector) + r2 ;
              if(delta < 0 )
              {
                  vector3 color(0,0,0); //fill this in with the appropriate colors
                  plot(x,y,color.x,color.y,color.z);
                  //~ Troll Code
                  //~ if(x % 5 == 0 || y % 5 == 0)
                  //~ {
                      //~ vector3 color(1, 1, 0); //fill this in with the appropriate colors
                      //~ plot(x,y,color.x,color.y,color.z);
                  //~ }

              }
              else if(delta == 0)
              {
                  if(intersectionOne >= 0)
                  {
                      vector3 color(1, 1, 1); //fill this in with the appropriate colors
                      plot(x,y,color.x,color.y,color.z);
                  }
              }
              else if(delta >= 0)
              {
                  if(intersectionOne >= 0)
                  {
                      vector3 color(1, 1, 1); //fill this in with the appropriate colors
                      plot(x,y,color.x,color.y,color.z);
                      
                  }
                  else if(intersectionOne < 0 && intersectionTwo >= 0)
                  {
                      vector3 color(1, 1, 1); //fill this in with the appropriate colors
                      plot(x,y,color.x,color.y,color.z);
                  }
                  else if(intersectionOne < 0 && intersectionTwo < 0)
                  {
                      vector3 color(0,0,0); //fill this in with the appropriate colors
                      plot(x,y,color.x,color.y,color.z);
                  }
              }
          }
      }
      SDL_GL_SwapBuffers(); //Finished drawing
  }
  //Exit out
  SDL_Quit();
  return 0;
}
*/

#include "ray_tracer.h"

using namespace std;

#define SET_RED(P, C)   (P = (((P) & 0x00ffffff) | ((C) << 24)))
#define SET_GREEN(P, C) (P = (((P) & 0xff00ffff) | ((C) << 16)))
#define SET_BLUE(P, C)  (P = (((P) & 0xffff00ff) | ((C) << 8)))
//--------------------------------------------------------------------------------
//Constants
//--------------------------------------------------------------------------------

const double            Object::small_t=1e-6;
const double            T_MAX = FLT_MAX;
int                     MAX_RECURSIVE_DEPTH = 4;
const Vector_3D<double> DEFAULT_COLOR = Vector_3D<double>(0.9961f, 0.7725f, 0.5451f);
const Vector_3D<double> BLACK = Vector_3D<double>(0.0f, 0.0f, 0.0f);
const Vector_3D<double> WHITE = Vector_3D<double>(1.0f, 1.0f, 1.0f);

//--------------------------------------------------------------------------------
//Utility Functions
//--------------------------------------------------------------------------------
double sqr(const double x)
{
  return x*x;
}

Pixel Pixel_Color(const Vector_3D<double>& color)
{
  Pixel pixel=0;
  SET_RED(pixel,(unsigned char)(min(color.x,1.0)*255));
  SET_GREEN(pixel,(unsigned char)(min(color.y,1.0)*255));
  SET_BLUE(pixel,(unsigned char)(min(color.z,1.0)*255));
  return pixel;
}

//--------------------------------------------------------------------------------
//Shaders
//--------------------------------------------------------------------------------
Vector_3D<double> Flat_Shader::Shade_Surface(const Ray& ray,const Object& intersection_object,const Vector_3D<double>& intersection_point,const Vector_3D<double>& same_side_normal) const
{
  return color;
}

Vector_3D<double> Phong_Shader::Shade_Surface(const Ray& ray,const Object& intersection_object,const Vector_3D<double>& intersection_point,const Vector_3D<double>& same_side_normal) const
{
  /*
    I = Ia + Id + Is = Ra * La + Rd * Ld * max(0, 1 * n) + Rs * Ls * pow(max(0, cos(theta)), alpha)
    I = Color Intensity
    R = Reflectance
    L = Illumination
  */  
    
  //Intensity
  Vector_3D<double> color; 
  
  //Calculate color from each light source
  for(unsigned i = 0; i < world.lights.size(); i++)
  {
    Light* currentLight = world.lights[i];
    Vector_3D<double> currentPosition(currentLight->position);
    Vector_3D<double> illumination(currentLight->Emitted_Light(ray));
    Vector_3D<double> intersection(intersection_point + (same_side_normal * intersection_object.small_t));


    //Ambient Lighting
    Vector_3D<double> ambientLight(color_ambient * illumination);

    //Diffuse Lighting
    Vector_3D<double> L(currentPosition - intersection);
    L.Normalize();
    double LN = Vector_3D<double>::Dot_Product(L, same_side_normal);
    Vector_3D<double> diffuseLight(color_diffuse * illumination * max(0.0, LN));    

    //Specular Lighting
    
    //View Direction
    Vector_3D<double> V(ray.endpoint - intersection);
    V.Normalize();
    
    //Reflected Direction
    Vector_3D<double> R(((same_side_normal * LN) * 2.0) - L);
    R.Normalize();
    
    double VR = Vector_3D<double>::Dot_Product(V, R);
    Vector_3D<double> specularLight(color_specular * illumination * pow(max(0.0, VR), specular_power));

    //Shadows are enabled
    if(world.enable_shadows == true)
    {
      Ray shadow(intersection, currentPosition - intersection);
      for(unsigned j = 0; j < world.objects.size(); j++)
      {
        if(world.objects[j]->Intersection(shadow) == false)
        {
          //Not sure why this gives the desired color
          color += diffuseLight + specularLight;
        }
      }
      
      //Playing around with lighting
      color += ambientLight * diffuseLight * specularLight;
    }
    
    //Shadows are not enabled
    else
    {
      //Colors are the sum of the ambient, diffuse, and specular lighting
      color += ambientLight + diffuseLight + specularLight;
    }
  }
  
  //return the color
  return color;
}

Vector_3D<double> Reflective_Shader::Shade_Surface(const Ray& ray,const Object& intersection_object,const Vector_3D<double>& intersection_point,const Vector_3D<double>& same_side_normal) const
{
  //~ http://paulbourke.net/geometry/reflected/
  
  //Base Color for Sphere
  Vector_3D<double> sphereColor(Phong_Shader::Shade_Surface(ray, intersection_object, intersection_point, same_side_normal));
  
  //Adjusting the intersection point so that it doesn't intersect with itself
  Vector_3D<double> intersection(intersection_point + (same_side_normal * intersection_object.small_t));
  
  //Calculating the direction of the intersection
  Vector_3D<double> intersectionDirection(intersection_point - ray.endpoint);
  
  //Finding the reflected ray direction
  Vector_3D<double> reflectedRayDirection(intersectionDirection - same_side_normal * 2 * Vector_3D<double>::Dot_Product(intersectionDirection, same_side_normal));
  
  //A ray just for place holder reasons for a function
  Ray dummy_ray;
  
  //Create ray with intersection we calculated and direction
  Ray reflectedRay(intersection, reflectedRayDirection);
  
  //Cast the ray we created
  Vector_3D<double> reflectedColor(world.Cast_Ray(reflectedRay, dummy_ray));
  
  //Calculate the reflected intensity added to the base intensity
  Vector_3D<double> color(reflectedColor * reflectivity + sphereColor);
  
  return color;
}

//--------------------------------------------------------------------------------
//Object Ray Interaction
//--------------------------------------------------------------------------------
//Return the plane normal at location
Vector_3D<double> Plane::Normal(const Vector_3D<double>& location) const
{
  return normal;
}

//Normalize the coordinates at location
Vector_3D<double> Sphere::Normal(const Vector_3D<double>& location) const
{
  Vector_3D<double> normal(location - center);
  normal.Normalize(); 
  return normal;
}

//Determine if the ray intersects with the sphere
//If there is an intersection, set t_max, current_object, and semi_infinite as appropriate and return true
bool Sphere::Intersection(Ray& ray) const
{
  //~* http://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection
  //~* http://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/minimal-ray-tracer-rendering-spheres
  Vector_3D<double> cameraView(ray.direction);
  Vector_3D<double> cameraDirection(ray.endpoint - center);
  double rSquared = radius * radius;

  double a = Vector_3D<double>::Dot_Product(cameraView, cameraView);
  double b = 2 * Vector_3D<double>::Dot_Product(cameraView, cameraDirection);
  double c = Vector_3D<double>::Dot_Product(cameraDirection, cameraDirection) - rSquared;
  
  double discriminant = b * b - 4 * a * c;
  
  //In case that there are 2 non-real intersections
  if(discriminant < 0)
  {
    return false;
  }
  else
  {
    //To store the point of closest intersection from Ray-Sphere intersection
    double t = 0;
    
    double intersectionOne = (-1.0 * b + sqrt(discriminant) ) / (2.0 * a);
    double intersectionTwo = (-1.0 * b - sqrt(discriminant) ) / (2.0 * a);
    
    //If IntersectionOne is real but IntersectionTwo is not
    if(intersectionOne >= 0 && intersectionTwo < 0)
    { 
      //Take intersectionOne as new distance t
      t = intersectionOne;
    }
    
    //If IntersectionTwo is real but IntersectionOne is not
    else if( intersectionTwo >= 0 && intersectionOne < 0 ) 
    {
      //Take intersectionTwo as new distance t
      t = intersectionTwo;
    }
    
    //If IntersectionOne is real and IntersectionTwo is real
    else if( intersectionOne >= 0 && intersectionTwo >= 0 ) 
    {
      //Take minimum of intersectionOne and intersectionTwo as new distance t
      t = min(intersectionOne, intersectionTwo);
    }
    else
    {
      return false;
    }

    //If there is a closer intersection than the one already set and value is not super small
    if(ray.semi_infinite == true && t > small_t)
    {
      ray.current_object = this;
      ray.t_max = t;
      ray.semi_infinite = false;
      return true;
    }
    else if(ray.semi_infinite == false  && t < ray.t_max  && t>small_t)
    {
      ray.current_object = this;
      ray.t_max = t;
      return true;
    }    
    return false;
  }
  return false;
}

//Determine if the ray intersects with the sphere
//If there is an intersection, set t_max, current_object, and semi_infinite as appropriate and return true
bool Plane::Intersection(Ray& ray) const
{
  //~* http://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-plane-and-ray-disk-intersection
  //~ http://www.cs.princeton.edu/courses/archive/fall00/cs426/lectures/raycast/sld017.htm
  //~ http://stackoverflow.com/questions/23975555/how-to-do-ray-plane-intersection
  double numerator    = Vector_3D<double>::Dot_Product(normal, x1 - ray.endpoint);
  double denominator  = Vector_3D<double>::Dot_Product(normal, ray.direction);
  double t            = numerator / denominator;
  
  if(ray.semi_infinite == true && t > small_t)
  {
    ray.current_object = this;
    ray.t_max = t;
    ray.semi_infinite = false;
    return true;
  }
  else if(ray.semi_infinite == false  && t < ray.t_max  && t>small_t)
  {
    ray.current_object = this;
    ray.t_max = t;
    return true;
  }    

  //Otherwise false
  return false;
}

//--------------------------------------------------------------------------------
//Camera
//--------------------------------------------------------------------------------
//Find the world position of the input pixel
Vector_3D<double> Camera::World_Position(const Vector_2D<int>& pixel_index)
{
  //Location of the pixel within the view of the camera grid
  Vector_2D<double> position = film.pixel_grid.X(pixel_index);
  
  //Location of the pixel within the world with the Z of focal point
  //Made the mistake here too where Vector_3D does not have = operator
  Vector_3D<double> result(focal_point + horizontal_vector * position.x + vertical_vector * position.y);  
  
  return result;
}

//--------------------------------------------------------------------------------
//Render_World
//--------------------------------------------------------------------------------
//Find the closest object of intersection and return a pointer to it
//  If the ray intersects with an object, then ray.t_max, ray.current_object, and ray.semi_infinite will be set appropriately
//  If there is no intersection do not modify the ray and return 0
const Object* Render_World::Closest_Intersection(Ray& ray)
{
  bool intersection;
  for(unsigned i = 0; i < objects.size(); i++)
  {
    //Object intersection data set into ray if intersection exists.
    //If not nothing is changed from ray's data.
    if(objects[i]->Intersection(ray) == true)
    {
      intersection = true;
    }
  }
  
  //If an object is found, it returns the object
  if(intersection == true)
  {
    return ray.current_object;
  }
  //Else it returns NULL
  else
  {
    return NULL;
  }
}

//set up the initial view ray and call 
void Render_World::Render_Pixel(const Vector_2D<int>& pixel_index)
{
  Ray dummy_root;
  
  //Set ray start to camera position, Set direction of ray to pixel from camera
  //Made the mistake of using = operator ray = Ray(...) 
  Ray ray(camera.position, camera.World_Position(pixel_index) - camera.position); 
  
  //Set max distance so that the ray can travel
  ray.t_max = T_MAX;
  
  Vector_3D<double> color = Cast_Ray(ray, dummy_root);
  camera.film.Set_Pixel(pixel_index, Pixel_Color(color));
}

//Cast ray and return the color of the closest intersected surface point, 
//Or the background color if there is no object intersection
Vector_3D<double> Render_World::Cast_Ray(Ray& ray,const Ray& parent_ray)
{
  //Create
  Vector_3D<double> color(BLACK);

  const Object * possibleObject = Closest_Intersection(ray);
  
  //Determine the color here
  if(possibleObject != NULL)
  {
    Vector_3D<double> pointOfIntersection = ray.Point(ray.t_max);
    Vector_3D<double> normalToObject      = possibleObject->Normal(pointOfIntersection);
    
    //Return the color of the object
    return possibleObject->material_shader->Shade_Surface(ray, *possibleObject, pointOfIntersection, normalToObject);
  }
  
  //Return black otherwise
  return color;
}
