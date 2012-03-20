uniform sampler2D phongTex;
uniform sampler2D cocTex;

uniform float texDx;
uniform float DOF;
varying vec2 texcoord;

float distance(vec2 tcoords, vec2 uv)
{
    float dx = tcoords.x - uv.x;
    float dy = tcoords.y - uv.y;
    return sqrt(dx*dx + dy*dy);
}

vec3 gaussianBlur(sampler2D tex, float n)
{
    vec3 sum = vec3(0.0);
    float v = texcoord.y - n/2.0 * texDx;
    float totWeight = 0.0;
    for (int i = 0; i < n; ++i, v += texDx){
        float d = distance(texcoord, vec2(texcoord.x,v ));
        float weight = exp(-d*d);
        sum += weight*texture2D(tex, vec2(texcoord.x,v)).rgb;
        totWeight += weight;
    }
    return sum / totWeight;
}

void main() {
    float t = abs(texture2D(cocTex,texcoord).r-0.5)*2;
	//First color buffer
    float blurAmount = t*t*t*t;
	gl_FragColor = vec4(gaussianBlur(phongTex,blurAmount*DOF),1);
    //gl_FragColor = vec4(1,0,0,1);
}
