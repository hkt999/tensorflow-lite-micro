#ifndef _NMS_ALGORITHM_H_
#define _NMS_ALGORITHM_H_

#include "nms.h"

class BoundingBox {
    public:
        int minX;
        int minY;
        int maxX;
        int maxY;
        int score;
        int classId;

    public:
        BoundingBox() {} // default constructor
        BoundingBox(int minX, int minY, int maxX, int maxY, int score, int classId);
        BoundingBox( BoundingBox &c) { *this = c; }
        virtual ~BoundingBox() {}
        BoundingBox &operator= (BoundingBox &box);
        BoundingBox &operator*= (BoundingBox &box);
        BoundingBox &operator+= (BoundingBox &box);
        bool operator<(BoundingBox &box);

        inline int GetClassId() { return classId; }

        int Area();
        int IoU(BoundingBox &input); /* 100% ratio */
};

#define MAX_BOXES   10

class NmsCb {
    public:
        NmsCb() {}
        virtual ~NmsCb() {}

    public:
        virtual int callback(BoundingBox &boundingBox) = 0;
};

class ImageClass {
    protected:
        BoundingBox boxArray[MAX_BOXES];
        BoundingBox pickArray[MAX_BOXES];
        int numBox;
        int numPicked;
        int classId;

    public:
        ImageClass() { numBox = 0; numPicked = 0; classId = -1; }
        virtual ~ImageClass() {}
        int AddBoundingBox( BoundingBox &box );

        inline int GetClassId() { return classId; }
        inline void SetClassId(int id) { classId = id; }

        void Dump(NmsCb &cb);

    public:
        void SortBoxes();
        void Go(int overlayThreshold);

    friend class NmsPostPorcess;
};

class NmsPostProcess {
    protected:
        ImageClass imageClass[MAX_BOXES];
        int numClasses;

    public:
        NmsPostProcess() { numClasses = 0; }
        ~NmsPostProcess() {}

    public:
        int AddBoundingBox( BoundingBox &box );
        void Go(int overlayThreshold, NmsCb &cb);
};

#endif

