using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class CarAgent : Agent
{
    public GameObject CarPrefab;
    public float motorMax, steerMax;

    private GameObject car;
    private Vector3 forward;
    private WheelCollider fl, fr, hl, hr;

    public override void InitializeAgent()
    {
        AgentReset();
    }

    public override void CollectObservations()
    {
        AddVectorObs(Vector3.Dot(forward, car.GetComponent<Rigidbody>().velocity));
        
    }

    public override void AgentAction(float[] vectorAction, string textAction)
    {
        float angle_z = car.transform.rotation.eulerAngles.z > 180 ? 360 - car.transform.rotation.eulerAngles.z : car.transform.rotation.eulerAngles.z;
        if (Mathf.Abs(angle_z) > 10)
        {
            SetReward(0);
            Done();
            return;
        }

        if (car.transform.position.y < -0.5f)
        {
            SetReward(-50);
            Done();
            return;
        }
        float motor = -vectorAction[0];
        float steer = vectorAction[1] * steerMax;
        hl.motorTorque = motor * motorMax;
        hr.motorTorque = motor * motorMax;
        Vector3 position;
        Quaternion rotation;

        fl.steerAngle = steer;
        fr.steerAngle = steer;
        fl.GetWorldPose(out position, out rotation);
        fl.transform.rotation = rotation;
        fr.transform.rotation = rotation;
        hr.GetWorldPose(out position, out rotation);
        hl.transform.rotation = rotation;
        hr.transform.rotation = rotation;
        forward = Vector3.Normalize(fl.transform.position - hl.transform.position);
        SetReward(Vector3.Dot(forward, car.GetComponent<Rigidbody>().velocity) /5);
        if (GetReward() < 0)
        {
            float reward = GetReward();
            reward *= 4;
            SetReward(reward);
        }
    }

    public override void AgentReset()
    {
        DestroyImmediate(car);
        car = Instantiate(CarPrefab, new Vector3(5f, 1f, 0f), Quaternion.Euler(0f, 0f, 0f));
        car.GetComponent<Rigidbody>().velocity = Vector3.zero;
        agentParameters.agentCameras[0] = car.transform.Find("Camera").GetComponent<Camera>();
        fl = car.transform.Find("Alloys01").Find("fl").GetComponent<WheelCollider>();
        fr = car.transform.Find("Alloys01").Find("fr").GetComponent<WheelCollider>();
        hl = car.transform.Find("Alloys01").Find("hl").GetComponent<WheelCollider>();
        hr = car.transform.Find("Alloys01").Find("hr").GetComponent<WheelCollider>();
        forward = Vector3.Normalize(fl.transform.position - hl.transform.position);
        car.SetActive(true);
    }

    public override void AgentOnDone()
    {
        AgentReset();
    }

}
