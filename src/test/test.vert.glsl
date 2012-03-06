uniform float pointRadius;  
uniform float pointScale;  
void main()
{
    vec3 pos = (gl_ModelViewMatrix * gl_Vertex).xyz; 
    float dist = 2.0;//length(pos);
    gl_PointSize = pointRadius * (pointScale / dist);
    gl_TexCoord[0] = gl_MultiTexCoord0;
    gl_Position = ftransform();
    gl_FrontColor = gl_Color;
}