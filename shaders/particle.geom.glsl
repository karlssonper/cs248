#version 330
#extension GL_EXT_geometry_shader4 : enable

layout(points) in;
layout (triangle_strip, max_vertices=4) out;

uniform mat4 ProjectionMatrix;

in float time[];
in float size[];
out float timeCopy;
out vec2 texCoord;

void main() {

    gl_Position = ProjectionMatrix * (vec4(-size[0],-size[0],0.0,0.0) + gl_PositionIn[0]);
    timeCopy = time[0];
    texCoord = vec2(0.0, 0.0);
    EmitVertex();

    gl_Position = ProjectionMatrix * (vec4(size[0],-size[0],0.0,0.0) + gl_PositionIn[0]);
    timeCopy = time[0];
    texCoord = vec2(1.0, 0.0);
    EmitVertex();

    gl_Position = ProjectionMatrix * (vec4(-size[0],size[0],0.0,0.0) + gl_PositionIn[0]);
    timeCopy = time[0];
    texCoord = vec2(.0, 1.0);
    EmitVertex();

    gl_Position = ProjectionMatrix * (vec4(size[0],size[0],0.0,0.0) + gl_PositionIn[0]);
    timeCopy = time[0];
    texCoord = vec2(1.0, 1.0);
    EmitVertex();

    EndPrimitive();
}
