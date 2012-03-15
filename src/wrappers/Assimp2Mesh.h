/*
 * Assimp2Mesh.h
 *
 *  Created on: Mar 6, 2012
 *      Author: per
 */

#ifndef ASSIMP2MESH_H_
#define ASSIMP2MESH_H_

#ifdef _WIN32
    #include "assimp.hpp"
    #include "aiScene.h"
    #include "aiPostProcess.h"
    #include <memory>
    #include <iostream>
#else
    #include <Importer.hpp>
    #include <scene.h>
    #include <postprocess.h>
#endif

#include <Mesh.h>
#include <string>

namespace ASSIMP2MESH{

void read(std::string _fileStr,
          std::string _meshStr,
          Mesh * _mesh)
{
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(_fileStr.c_str(),
        aiProcess_CalcTangentSpace |
        aiProcess_Triangulate |
        aiProcess_JoinIdenticalVertices |
        aiProcessPreset_TargetRealtime_Quality);
    int idx = -1;
    for (unsigned int i = 0; i < scene->mNumMeshes; ++i){
        aiMesh * mesh = scene->mMeshes[i];
        if (std::string(mesh->mName.data) == _meshStr){
            std::cout << "Found " <<_meshStr << " in " << _fileStr << std::endl;
            idx = i;
            break;
        }
    }

    if (idx < 0) {
        std::cerr << "Could not find a mesh named " << _meshStr
                << " in " << _fileStr << std::endl;
        return;
    }

    aiMesh * mesh = scene->mMeshes[idx];

    std::vector<Vector3> position;
    std::vector<Vector2> texCoord;
    std::vector<Vector3> normal;
    std::vector<Vector3> tangent;
    std::vector<Vector3> bitangent;

    position.resize(mesh->mNumVertices);

    if (mesh->HasTextureCoords(0)) {
        texCoord.resize(mesh->mNumVertices);
    } else {
        texCoord.resize(0);
    }

    if (mesh->HasNormals()) {
        normal.resize(mesh->mNumVertices);
    } else {
        normal.resize(0);
    }

    if (mesh->HasTangentsAndBitangents()) {
        tangent.resize(mesh->mNumVertices);
        bitangent.resize(mesh->mNumVertices);
    }
    else {
        tangent.resize(0);
        bitangent.resize(0);
    }

    for (unsigned int j = 0; j < mesh->mNumVertices; ++j) {
        position[j].x = mesh->mVertices[j].x;
        position[j].y = mesh->mVertices[j].y;
        position[j].z = mesh->mVertices[j].z;

        if (mesh->HasTextureCoords(0)) {
            texCoord[j].x = mesh->mTextureCoords[0][j].x;
            texCoord[j].y = mesh->mTextureCoords[0][j].y;
        }

        if (mesh->HasNormals()) {
            normal[j].x = mesh->mNormals[j].x;
            normal[j].x = mesh->mNormals[j].y;
            normal[j].x = mesh->mNormals[j].z;
        }
        if (mesh->HasTangentsAndBitangents()){
            tangent[j].x = mesh->mTangents[j].x;
            tangent[j].y = mesh->mTangents[j].y;
            tangent[j].z = mesh->mTangents[j].z;
            bitangent[j].x = mesh->mBitangents[j].x;
            bitangent[j].y = mesh->mBitangents[j].y;
            bitangent[j].z = mesh->mBitangents[j].z;
        }
    }

    std::vector<unsigned int> indices;
    indices.reserve(mesh->mNumFaces * 3);
    for (unsigned int i = 0; i < mesh->mNumFaces; ++i) {
        if (mesh->mFaces[i].mNumIndices != 3) {
            //std::cerr << "Can't process nothing else than triangles"
                   // << " # verts: " <<  mesh->mFaces[i].mNumIndices << std::endl;
            continue;
        }
        indices.push_back(mesh->mFaces[i].mIndices[0]);
        indices.push_back(mesh->mFaces[i].mIndices[1]);
        indices.push_back(mesh->mFaces[i].mIndices[2]);
    }

    _mesh->geometryIs(position, texCoord, normal, tangent, bitangent,indices);


}

}//end of namespace

#endif /* ASSIMP2MESH_H_ */
