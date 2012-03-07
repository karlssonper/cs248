#include <stdio.h>
#include <stdlib.h>
#include <Graphics.h>
#include <GL/glut.h>
#include <wrappers/Assimp2Mesh.h>
#include <Node.h>
#include <Mesh.h>
#include <Camera.h>
#include <Graphics.h>

static int width = 600, height = 600;
Mesh * mesh;
Node * node;
Camera * cam;

static float randf()
{
    return (float) rand() / ((float) RAND_MAX + 1);
}

static void createSceneGraph()
{
    mesh = new Mesh("sixten");
    node = new Node("sixtenNode");
    cam = new Camera();
    cam->projectionIs(45.f, 1.f, 1.f, 100.f);

    ASSIMP2MESH::read("../models/armadillo.3ds", "0", mesh);
    std::string tex("../textures/armadillo_n.jpg");
    Graphics::instance().texture(tex);
    //Graphics::instance().deleteTexture(tex);

    std::string shader("../shaders/phong");
    Graphics::instance().shader(shader);
    //Graphics::instance().deleteShader(shader);
}

static void display(void)
{
    glClearColor(randf(), randf(), randf(), 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    GLuint _vao;
    glGenVertexArrays(1, &_vao);
    glBindVertexArray(_vao);
    glutSwapBuffers();
    glutPostRedisplay();
}

static void reshape(int w, int h)
{
    width = w > 1 ? w : 1;
    height = h > 1 ? h : 1;
    glViewport(0, 0, width, height);
    glClearDepth(1.0);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glEnable(GL_DEPTH_TEST);
}

int main(int argc, char **argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("cookie");

    glutReshapeFunc(reshape);
    glutDisplayFunc(display);

    if (gl3wInit()) {
        fprintf(stderr, "failed to initialize OpenGL\n");
        return -1;
    }
    if (!gl3wIsSupported(3, 2)) {
        fprintf(stderr, "OpenGL 3.2 not supported\n");
        return -1;
    }
    printf("OpenGL %s, GLSL %s\n", glGetString(GL_VERSION),
           glGetString(GL_SHADING_LANGUAGE_VERSION));

    createSceneGraph();

    glutMainLoop();
    return 0;
}
