// Implementation of the templated Vector class
// ECE6122 Lab 5
// Seth Stewart

#include <iostream> // debugging
#include "Vector.h"
using namespace std;

// Default constructor
template <typename T>
Vector<T>::Vector()
{
  elements = NULL;
  count = 0;
  reserved = 0;
}

// Copy constructor
template <typename T>
Vector<T>::Vector(const Vector& rhs)
{
  //copy the values of the input vector rhs
  count = rhs.count;
  reserved = rhs.reserved;
  elements = (T*)malloc(reserved * sizeof(T));
  for(int i=0; i < reserved; i++)
  {
    new (&elements[i]) T(rhs.elements[i]);
  }
}

// Assignment operator
template <typename T>
Vector<T>& Vector<T>::operator=(const Vector& rhs)
{
  if (! (rhs == this))
  {
	  this.clear();
  }
	count = rhs.count;
	reserved = rhs.reserved;
	elements = (T*)malloc(reserved * sizeof(T));
	for(int i=0; i < count; i++)
	{
		new (&elements[i]) T(rhs.elements[i]);
	}
  
}

#ifdef GRAD_STUDENT
// Other constructors
template <typename T>
Vector<T>::Vector(size_t nReserved)
{ 
  // Initialize with reserved memory
  reserved = nReserved;
  count = 0;
  elements = (T*)malloc(reserved *sizeof(T));
}

template <typename T>
Vector<T>::Vector(size_t n, const T& t)
{ // Initialize with "n" copies of "t"
 elements = (T*)malloc(n*sizeof(T));
 for(int i=0; i < n; i++)
 {
   new(&elements[i])T(t);
 }
 count = n; 
 reserved = n;
}
#endif

// Destructor
template <typename T>
Vector<T>::~Vector()
{
  for (int i =0; i < count; i++)
  {
    elements[i].~T(); 
  }
  free(elements);
  count = 0;
}

// Add and access front and back
template <typename T>
void Vector<T>::Push_Back(const T& rhs)
{
  if (count < reserved) //if space is already reserved
  {
    new (&elements[count])T(rhs); //place it after the last entry
    count++;
  }
  else
  {
    T* newPointer = (T*)malloc((count+1)*sizeof(T)); //make space for it in a new pointer
    for (int i=0; i < count; i++) //for every entry in the old vector
    {
      new (&newPointer[i])T(elements[i]); //inline copy to the new vector
      elements[i].~T(); //destroy that entry
    }
    free(elements); //free the memory
    elements = newPointer; //set the elements pointer to the new vector pointer
    new (&elements[count])T(rhs); //tack on the new element at the end
    count++;
    reserved = count;
  }
}

template <typename T>
void Vector<T>::Push_Front(const T& rhs)
{
  if(count < reserved) //if the current vector has room
  {   
    for (int i=0; i < count; i++)
    {
      elements[i+1] = elements[i];
    }
    new (&elements[0])T(rhs);
  }
  else
  {
    T* newPointer= (T*)malloc((count + 1) * sizeof(T));  //make a new space for it otherwise

    for(int i = 0; i < count; i++)
    {
      new (&newPointer[i+1])T(elements[i]); //move each element over
      elements[i].~T(); // destroy the old element
    }
    free(elements); // free the memory 
    elements = newPointer; //reassign the pointer to the new pointer
    new (&elements[0])T(rhs); //tack on the new element to the front
    count ++;        
    reserved = count;
  }
}

template <typename T>
void Vector<T>::Pop_Back()
{ // Remove last element (and decrement count)
  elements[count-1].~T();
  count--;
}

template <typename T>
void Vector<T>::Pop_Front()
{ // Remove first element (and decrement count)(and fix the other entries)
  elements[0].~T(); //destroy the first element
  for(int i=0; i< count-1; i++) 
  {
    new(&elements[i])T(elements[i+1]); //move element at i+1 to i
    elements[i+1].~T(); //destroy the element just moved
  }
  count--;
}

// Element Access
template <typename T>
T& Vector<T>::Front() const
{
  return elements[0]; //return the address of the first element
}

// Element Access
template <typename T>
T& Vector<T>::Back() const
{
  return elements[count-1];//return the address of the last element
}

template <typename T>
const T& Vector<T>::operator[](size_t i) const
{
  return elements[i];
}

template <typename T>
T& Vector<T>::operator[](size_t i) 
{
  return elements[i];
}

template <typename T>
size_t Vector<T>::Size() const
{
  return count;
}

template <typename T>
bool Vector<T>::Empty() const
{
  return count == 0;
}

// Implement clear
template <typename T>
void Vector<T>::Clear()
{
  //destroy the elements but dont free memory
  for (int i=0; i<count; i++)
  {
    elements[i].~T();
  }
  count = 0; //reset count but keep reserved space
}

// Iterator access functions
template <typename T>
VectorIterator<T> Vector<T>::Begin() const
{
  return VectorIterator<T>(elements);
}

template <typename T>
VectorIterator<T> Vector<T>::End() const
{
  return VectorIterator<T>(&elements[count]);
}

#ifdef GRAD_STUDENT
// Erase and insert
template <typename T>
void Vector<T>::Erase(const VectorIterator<T>& it)
{
  VectorIterator<T> iter;//make an itterator
  int pos=0;
  for (iter = elements; iter != (&elements[count]); iter++)
  {
    if (iter == it) break;
    pos++;
  }
  
  elements[pos].~T();
  
  for(int i = pos; i < count-1; i++)
  {
    new(&elements[i])T(elements[i+1]);
    elements[i+1].~T();
  }
  count--;
}

template <typename T>
void Vector<T>::Insert(const T& rhs, const VectorIterator<T>& it)
{ 
  //I was going to do an if count<reserved here, but the amount of code for the if statement is silly
  T* newPointer = (T*)malloc((count+1) * sizeof(T)); // so just make space, and assign a new pointer
  VectorIterator<T> iter; // find the 'it' position using an iterator
  int pos = 0;
  
  for(iter = elements; iter != (&elements[count]); iter++) //loop through the vector
  { 
    if(iter == it) break; // and find the position where 'it' is found 
    pos++;       
  } 
  
  for(int i = 0; i < pos; i++) //for the elements before 'it'
  {
    new (&newPointer[i])T(elements[i]);  // copy the values over to the new pointer
    elements[i].~T(); //and destroy them from the old pointer
  }
  
  new(&newPointer[pos])T(rhs);  //put rhs at the position
  
  for(int i = pos+1; i < count; i++) //copy the rest over
  {
    new (&newPointer[i])T(elements[i - 1]); //at their old posionts+1
    elements[i - 1].~T(); // and destroy old
  }

  free(elements); // free the memory held by the old pointer
  elements = newPointer; // elements pointer points to the newly allocated memory which was pointed by temp
  count ++;
}
#endif














// Implement the iterators

// Constructors
template <typename T>
VectorIterator<T>::VectorIterator()
{
  current = NULL;
}

template <typename T>
VectorIterator<T>::VectorIterator(T* c)
{
  current = c;
}

// Copy constructor
template <typename T>
VectorIterator<T>::VectorIterator(const VectorIterator<T>& rhs)
{
  current = rhs.current;
}

// Iterator defeferencing operator
template <typename T>
T& VectorIterator<T>::operator*() const
{
  return *current;
}

// Prefix increment
template <typename T>
VectorIterator<T>  VectorIterator<T>::operator++()
{
  current++;
  return *this;
}

// Postfix increment
template <typename T>
VectorIterator<T> VectorIterator<T>::operator++(int)
{
  VectorIterator<T> copy(*this);
  current++;
  return copy;
}

// Comparison operators
template <typename T>
bool VectorIterator<T>::operator !=(const VectorIterator<T>& rhs) const
{
  return current != rhs.current;
}

template <typename T>
bool VectorIterator<T>::operator ==(const VectorIterator<T>& rhs) const
{
  return current == rhs.current;
}