#include "intersections.h"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside)
    {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}


__host__ __device__ bool aabbIntersectionTest(
    glm::vec3 aabbMin,
    glm::vec3 aabbMax,
    Ray r)
{
    // slab test; a zero direction component gives +-inf, which min/max handle fine
    glm::vec3 invDir = 1.0f / r.direction;
    glm::vec3 t0 = (aabbMin - r.origin) * invDir;
    glm::vec3 t1 = (aabbMax - r.origin) * invDir;
    glm::vec3 tNear = glm::min(t0, t1);
    glm::vec3 tFar = glm::max(t0, t1);

    // enter = latest slab entry, exit = earliest slab exit
    float tEnter = glm::max(glm::max(tNear.x, tNear.y), tNear.z);
    float tExit = glm::min(glm::min(tFar.x, tFar.y), tFar.z);
    // tExit > 0 (not tEnter) so a ray starting inside the box still counts
    return tEnter <= tExit && tExit > 0.0f;
}

__host__ __device__ float triangleIntersectionTest(
    const Triangle& triangle,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    // glm 0.9.x: bary.x/.y weight vertices[1]/[2], bary.z is t; back faces are culled
    glm::vec3 bary;
    bool hit = glm::intersectRayTriangle(r.origin, r.direction, triangle.vertices[0], triangle.vertices[1], triangle.vertices[2], bary);
    // bary is unwritten on a miss, so check before reading it
    if (!hit || bary.z <= 0.0f) return -1.0f;

    float t = bary.z;
    intersectionPoint = getPointOnRay(r, t);

    // smooth shading: interpolate vertex normals with the barycentric weights
    float w0 = 1 - bary.x - bary.y;
    float w1 = bary.x;
    float w2 = bary.y;
    normal = glm::normalize(w0 * triangle.normals[0] + w1 * triangle.normals[1] + w2 * triangle.normals[2]);
    outside = dot(r.direction, normal) < 0;
    return t;
}
