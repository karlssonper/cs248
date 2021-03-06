uniform sampler2D phongTex;
uniform sampler2D cocTex;
uniform sampler2D hudTex;

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
        if (texture2D(cocTex,texcoord).r > 0.25) {
            float d = distance(texcoord, vec2(texcoord.x,v ));
            float weight = exp(-d*d);
            sum += weight*texture2D(tex, vec2(texcoord.x,v)).rgb;
            totWeight += weight;
        }
    }
    return sum / totWeight;
}

void main() {

    float blurAmount = texture2D(cocTex,texcoord).r;
    vec3 color;
    if (blurAmount > 0.25 && DOF != 0.0)
        color = gaussianBlur(phongTex,blurAmount*DOF);
    else
        color = texture2D(phongTex,texcoord).rgb;

    vec4 hud = texture2D(hudTex, texcoord);
    vec3 final = (1.0-hud.a)*color + hud.a*(hud.rgb);

    gl_FragColor = vec4(final,1.0);
}
