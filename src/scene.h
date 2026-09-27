#pragma once

#include "sceneStructs.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void loadGLTF(const std::string& path, Geom& geom);
    int loadTexture(const std::string& path);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
    std::vector<Triangle> triangles;
    std::vector<glm::vec3> texels;        // every image's pixels, back to back
    std::vector<TextureInfo> textures;    // where each image sits inside texels
};
