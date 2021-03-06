#include <stdio.h>
#include <stdlib.h>
#include <wrappers/Assimp2Mesh.h>
#include <Node.h>
#include <Mesh.h>
#include <Camera.h>
#include <Graphics.h>
#include <GL/glut.h>

static int width = 600, height = 600;
Mesh * mesh;
ShaderData * shader;
Node * node;

int mouseX, mouseY;

static float randf()
{
    return (float) rand() / ((float) RAND_MAX + 1);
}

static void createSceneGraph()
{
    node = new Node("sixtenNode");
    mesh = new Mesh("sixten", node);
    Camera::instance().projectionIs(45.f, 1.f, 1.f, 100.f);
    shader = new ShaderData("../shaders/phong");

    shader->enableMatrix(MODELVIEW);
    shader->enableMatrix(PROJECTION);
    shader->enableMatrix(NORMAL);

    std::string tex("../textures/armadillo_n.jpg");
    std::string texName("normalMap");
    shader->addTexture(texName, tex);

    mesh->shaderDataIs(shader);
    ASSIMP2MESH::read("../models/armadillo.3ds", "0", mesh, 1.0f);



    //Graphics::instance().deleteTexture(tex);

    //Graphics::instance().deleteShader(shader);
}

static void display(void)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    Camera::instance().BuildViewMatrix();
    mesh->display();
    glutSwapBuffers();
    glutPostRedisplay();
}

static void processNormalKeys(unsigned char key, int x, int y) {
    switch (key){
        case 27:
            exit(0);
        case 'w':
            Camera::instance().move(0.5);
            break;
        case 's':
            Camera::instance().move(-0.5);
            break;
        case 'a':
            Camera::instance().strafe(-0.5);
            break;
        case 'd':
            Camera::instance().strafe(0.5);
            break;
    }
}
void mouseFunc(int x,int y)
{
    int dx = x - mouseX;
    int dy = y - mouseY;
    mouseX = x;
    mouseY = y;
    Camera::instance().yaw(1.6*dx);
    Camera::instance().pitch(1.6*dy);
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
    glutCreateWindow("tests/graphics");

    glutReshapeFunc(reshape);
    glutDisplayFunc(display);
    glutKeyboardFunc(processNormalKeys);
    glutMotionFunc(mouseFunc);

    createSceneGraph();

    glutMainLoop();
    return 0;
}
