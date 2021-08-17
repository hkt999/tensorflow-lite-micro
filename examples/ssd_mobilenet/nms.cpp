#include "nms.h"
#include <cstdio>

extern const char* kCategoryLabels[];

#define MIN(a,b) ((a<b)?a:b)
#define MAX(a,b) ((a<b)?b:a)

BoundingBox::BoundingBox(int minX, int minY, int maxX, int maxY, int score, int classId):
    minX(minX), minY(minY), maxX(maxX), maxY(maxY), score(score), classId(classId)
{
}

BoundingBox &BoundingBox::operator=(BoundingBox &input)
{
    minX = input.minX;
    minY = input.minY;
    maxX = input.maxX;
    maxY = input.maxY;
    score = input.score;
    classId = input.classId;

    return *this;
}

BoundingBox &BoundingBox::operator*=(BoundingBox &input)
{
    minX = MAX(minX, input.minX);
    minY = MAX(minY, input.minY);
    maxX = MIN(maxX, input.maxX);
    maxY = MIN(maxY, input.maxY);

    return *this;
}

BoundingBox &BoundingBox::operator+=(BoundingBox &input)
{
    minX = MIN(minX, input.minX);
    minY = MIN(minY, input.minY);
    maxX = MAX(maxX, input.maxX);
    maxY = MAX(maxY, input.maxY);

    return *this;
}

int BoundingBox::Area()
{
    return (maxX - minX + 1 ) * (maxY - minY + 1 );
}

int BoundingBox::IoU(BoundingBox &input)
{
    BoundingBox inter(*this);
    inter *= input;
    int interArea = inter.Area();

    return (interArea * 100) / (Area() + input.Area() - interArea); /* IoU */
}

bool BoundingBox::operator<(BoundingBox &box)
{
    return score < box.score;
}


int ImageClass::AddBoundingBox( BoundingBox &box )
{
    boxArray[numBox ++ ] = box;

    return 0;
}


void ImageClass::SortBoxes()
{
    // bubble sort
    for (int i=0; i<numBox-1; i++) {
        for (int j=i+1; j<numBox; j++) {
            if (boxArray[i] < boxArray[j]) {
                BoundingBox t = boxArray[i];
                boxArray[i] = boxArray[j];
                boxArray[j] = t;
            }
        }
    }
}

void ImageClass::Go(int overlayThreshold)
{
    SortBoxes();

    int pivot = 0;
    while (pivot < numBox) {
        BoundingBox pickedBox = boxArray[pivot++];
        while (pivot < numBox) {
            if (pickedBox.IoU( boxArray[pivot] ) > overlayThreshold) {
                pickedBox += boxArray[pivot++];
            } else 
                break;
        }
        pickArray[numPicked++] = pickedBox;
    }
}

void ImageClass::Dump(NmsCb &cb)
{
    for (int i=0; i<numPicked; i++) {
        BoundingBox b = pickArray[i];
        cb.callback(b);
    }
}

int NmsPostProcess::AddBoundingBox( BoundingBox &box )
{
    for (int i=0; i<numClasses; i++) {
        if (imageClass[i].GetClassId() == box.GetClassId()) {
            imageClass[i].AddBoundingBox(box);
            return 0;
        }
    }

    if ( numClasses >= MAX_BOXES )
        return -1; /* fail */

    imageClass[ numClasses ].SetClassId( box.GetClassId());
    imageClass[ numClasses ].AddBoundingBox( box );
    numClasses++;

    return 0;
}

void NmsPostProcess::Go(int overlayThreshold, NmsCb &cb)
{
    for (int i=0; i<numClasses; i++) {
        imageClass[i].Go(overlayThreshold);
        imageClass[i].Dump(cb);
    }
}
