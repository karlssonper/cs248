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

void readNode(Mesh * _mesh,
              const aiScene * _scene,
              aiNode * _node,    
              std::vector<Vector3> * _position,
              std::vector<Vector2> * _texCoord,
              std::vector<Vector3> * _normal,
              std::vector<Vector3> * _tangent,
              std::vector<Vector3> * _bitangent,
              std::vector<unsigned int> * _indices,
              Matrix4 _parentTransform) 
{

    std::cout << "readNode reading node " << _node->mName.data << std::endl;

    Matrix4 nodeTransform = _parentTransform;

    aiMatrix4x4 transform(_node->mTransformation);
    transform.Transpose(); // opengl uses column major!

    // convert to Matrix4 format
    Matrix4 parentTransform; 
    parentTransform.m_[0] = transform.a1;
    parentTransform.m_[1] = transform.a2;
    parentTransform.m_[2] = transform.a3;
    parentTransform.m_[3] = transform.a4;
    parentTransform.m_[4] = transform.b1;
    parentTransform.m_[5] = transform.b2;
    parentTransform.m_[6] = transform.b3;
    parentTransform.m_[7] = transform.b4;
    parentTransform.m_[8] = transform.c1;
    parentTransform.m_[9] = transform.c2;
    parentTransform.m_[10] = transform.c3;
    parentTransform.m_[11] = transform.c4;
    parentTransform.m_[12] = transform.d1;
    parentTransform.m_[13] = transform.d2;
    parentTransform.m_[14] = transform.d3;
    parentTransform.m_[15] = transform.d4;

    nodeTransform = nodeTransform*parentTransform;

    std::cout << "Node " << _node->mName.data << " has " << 
        _node->mNumMeshes << " meshes" << std::endl;

    for (unsigned int i=0; i<_node->mNumMeshes; ++i) {

        aiMesh * mesh = _scene->mMeshes[_node->mMeshes[i]];
        std::cout << "Processing mesh " << mesh->mName.data << std::endl;
        std::cout << "Mesh has " << mesh->mNumVertices << " vertices " << std::endl;
        std::cout << "Current position array size: " << _position->size() << std::endl;

        unsigned int indexOffset = _position->size();

        for (unsigned int j=0; j<mesh->mNumVertices; ++j) {

            Vector3 position;
            position.x = mesh->mVertices[j].x;
            position.y = mesh->mVertices[j].y;
            position.z = mesh->mVertices[j].z;
            Vector3 transformedPosition = nodeTransform*position;
            _position->push_back(transformedPosition);

            if (mesh->HasTextureCoords(0)) {
                Vector2 texCoord;
                texCoord.x = mesh->mTextureCoords[0][j].x;
                texCoord.y = mesh->mTextureCoords[0][j].y;
                _texCoord->push_back(texCoord);
            }

            if (mesh->HasNormals()) {
                Vector3 normal;
                normal.x = mesh->mNormals[j].x;
                normal.y = mesh->mNormals[j].y;
                normal.z = mesh->mNormals[j].z;
                Vector3 transformedNormal = nodeTransform*normal;
                _normal->push_back(transformedNormal);
            }
            if (mesh->HasTangentsAndBitangents()) {
                Vector3 tangent, bitangent;
                tangent.x = mesh->mTangents[j].x;
                tangent.y = mesh->mTangents[j].y;
                tangent.z = mesh->mTangents[j].z;
                bitangent.x = mesh->mBitangents[j].x;
                bitangent.y = mesh->mBitangents[j].y;
                bitangent.z = mesh->mBitangents[j].z;
                Vector3 transformedTangent = nodeTransform*tangent;
                Vector3 transformedBitangent = nodeTransform*bitangent;
                _tangent->push_back(transformedTangent);
                _bitangent->push_back(transformedBitangent);
            }

        } // for numVertices

        
        for (unsigned int i = 0; i < mesh->mNumFaces; ++i) {
            if (mesh->mFaces[i].mNumIndices != 3) continue;
            _indices->push_back(indexOffset + mesh->mFaces[i].mIndices[0]);
            _indices->push_back(indexOffset + mesh->mFaces[i].mIndices[1]);
            _indices->push_back(indexOffset + mesh->mFaces[i].mIndices[2]);
        }

    } // for nummeshes

    for (unsigned int i=0; i<_node->mNumChildren; ++i) {
        readNode(_mesh, 
            _scene, 
            _node->mChildren[i], 
            _position,
            _texCoord,
            _normal,
            _tangent,
            _bitangent,
            _indices,
            nodeTransform);

    } // for children
}

void read(std::string _fileStr, std::string _meshStr, Mesh * _mesh, float _scale)
{
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(_fileStr.c_str(),
          aiProcess_CalcTangentSpace |
          aiProcess_Triangulate |
          aiProcess_JoinIdenticalVertices |
          aiProcessPreset_TargetRealtime_Quality);

    std::vector<Vector3> position;
    std::vector<Vector2> texCoord;
    std::vector<Vector3> normal;
    std::vector<Vector3> tangent;
    std::vector<Vector3> bitangent;
    std::vector<unsigned int> indices;

    Matrix4 transform = Matrix4::scale(_scale, _scale, _scale);
    
    readNode(_mesh, 
             scene,
             scene->mRootNode,
             &position,
             &texCoord,
             &normal,
             &tangent,
             &bitangent,
             &indices,
             transform);

    std::cout << "DONE!" << std::endl;
    std::cout << "position.size(): " << position.size() << std::endl;
   
    _mesh->geometryIs(position, texCoord, normal, tangent, bitangent, indices);

}

}//end of namespace

#endif /* ASSIMP2MESH_H_ */
